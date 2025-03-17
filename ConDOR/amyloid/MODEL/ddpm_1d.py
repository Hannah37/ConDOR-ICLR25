import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
# from ema_pytorch import EMA

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR

from denoising_diffusion_pytorch.version import __version__

import os
import sys
import wandb
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../'))
import ordinal_regression

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# data

class Dataset1D(Dataset):
    def __init__(self, args):
        super().__init__()
        self.dir = args.dir
        self.classes = args.classes

        self.min_age = args.age_min
        self.max_age = args.age_max

        self.data = []
        self.label = []
        self.age = []

        self.load_data()

    def load_data(self):
        self.max_visits = 0

        for filename in os.listdir(os.path.join(self.dir, 'data')):
            data = torch.load(os.path.join(self.dir, 'data', filename))
            data = data.unsqueeze(0) # num_channel = 1

            if data.shape[1] == 1:
                print(filename, 'The number of visit is 1')
                break
            self.data.append(data)

            label = pd.read_pickle(os.path.join(self.dir, 'label', filename[:-3]))
            label_num_lst = []
            for item in label:
                if item == 'CN':
                    label_num_lst.append(0)
                elif item == 'SMC':
                    label_num_lst.append(1)
                elif item == 'EMCI':
                    label_num_lst.append(2)
                elif item == 'LMCI':
                    label_num_lst.append(3)
                elif item == 'AD':
                    label_num_lst.append(4)
                else:
                    print('Label Error')

            '''one-hot'''
            subj_label = torch.tensor(label_num_lst)
            subj_label = F.one_hot(subj_label, num_classes=self.classes)
            self.label.append(subj_label) 

            if len(label) > self.max_visits: self.max_visits = len(label)

            age = pd.read_pickle(os.path.join(self.dir, 'age', filename[:-3]))
            age_ = [(round(a, 1) - self.min_age)/(self.max_age - self.min_age) for a in age]

            if (len(label) != len(age)) or (len(label) != data.shape[1]):
                print(filename, data.shape, len(label), len(age))
                print("Length of label, age, data should be same")

            assert (len(label) == len(age)) and (data.shape[1] == len(label)), "Length of label, age, data should be same"

            age_ = torch.tensor(age_)
            self.age.append(age_)

    def max_visit(self):
        return self.max_visits
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx] 
        label = self.label[idx]
        age = self.age[idx] 

        return data, label, age

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(idx, dim, dim_out = None):
    if idx == 0:
        return nn.Sequential(
            nn.Upsample(size=(37,), mode = 'nearest'),
            nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
        )
    
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
        )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class EmbedFC_age(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC_age, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x): # x.shape: #visit
        x = x.unsqueeze(-1) # x.shape: #visit x 1
        return self.model(x)

class EmbedFC_label(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC_label, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x): # x.shape: #visit x #classes
        return self.model(x)    

class EmbedFC_cond_mix(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC_cond_mix, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x): # x.shape: #visit x 2*cond_dim
        return self.model(x)    

# model

class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        n_classes = None,
        max_visit = None 
    ):
        super().__init__()

        # determine dimensions
        self.max_visit = max_visit

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        '''embed conditions'''
        cond_in = 32
        cond_out = 64
        self.age_embed1 = EmbedFC_age(1, cond_in)
        self.age_embed2 = EmbedFC_age(1, int(cond_in/2))
        self.label_embed1 = EmbedFC_label(n_classes, cond_in)
        self.label_embed2 = EmbedFC_label(n_classes, int(cond_in/2))
        self.cond_mix1 = EmbedFC_cond_mix(2*cond_in + 1, 2*cond_out)
        self.cond_mix2 = EmbedFC_cond_mix(cond_in + 1, cond_out)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in + 2*cond_out, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out + dim_in + cond_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(ind, dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding = 1) ## 여기서 36 됐음
                # Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding = 2)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, label, age, pos, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        age_emb1 = self.age_embed1(age) 
        age_emb2 = self.age_embed2(age) 
        label_emb1 = self.label_embed1(label.type(torch.float32))
        label_emb2 = self.label_embed2(label.type(torch.float32)) 
   
        cemb1 = torch.cat((age_emb1, label_emb1, pos), dim=-1) # 
        cemb2 = torch.cat((age_emb2, label_emb2, pos), dim=-1) # 

        cemb1 = self.cond_mix1(cemb1) # 
        cemb2 = self.cond_mix2(cemb2) # 

        assert not torch.isnan(cemb1).any().item(), "cemb1 has nan"
        assert not torch.isnan(cemb2).any().item(), "cemb2 has nan"
        assert not torch.isnan(x).any().item(), "x has nan"
        x = self.init_conv(x)
     
        assert not torch.isnan(x).any().item(), "x has nan"
        r = x.clone()

        t = self.time_mlp(time)

        h = [] # skip-connection

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)

            assert not torch.isnan(x).any().item(), "block1(x) has nan"

            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t) 

        for idx, (block1, block2, attn, upsample) in enumerate(self.ups):
         
            if cemb1.dim() == 1:
                cemb1 = cemb1.unsqueeze(0)
            cemb1_exp = cemb1.unsqueeze(2).expand(-1, -1, x.shape[-1])
         
            x = torch.cat((x, cemb1_exp, h.pop()), dim = 1)
            x = block1(x, t)

            if cemb2.dim() == 1:
                cemb2 = cemb2.unsqueeze(0)
            cemb2_exp = cemb2.unsqueeze(2).expand(-1, -1, x.shape[-1])
            x = torch.cat((x, cemb2_exp, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1) 

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        num_node,
        timesteps = 1000, # diffusion timestep T
        sampling_timesteps = None,
        objective = 'pred_noise',
        norm_min = 0.5,
        norm_max = 4.42,
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        args
    ):
        super().__init__()

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.args = args
        self.OR_model = ordinal_regression.OrdinalRegression(self.args)
    
        self.norm_min = norm_min
        self.norm_max = norm_max

        self.num_node = num_node 
        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.num_timesteps_fromv2 = 2

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps

        self.is_ddim_sampling = True 
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    def predict_start_from_noise(self, x_t, t, noise): 
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, label, age, pos, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, label, age, pos, x_self_cond)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise) 

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, label, age, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, label, age, t, x_self_cond)
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, label, age, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, label=label, age=age, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, init_data, label, age, shape):
        print('In the p_sample_loop() !!!!')
        batch, device = shape[0], self.betas.device

        data = init_data

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            data, x_start = self.p_sample(data, label, age, t, self_cond)

        return data

    @torch.no_grad()
    def ddim_sample_init(self, init_data, label, age, pos, shape, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        data = init_data
        
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            pred_noise, x_start, *_ = self.model_predictions(data, label, age, pos, time_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                data = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(data) 

            data = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        print('v1 last noise min max: ', torch.min(pred_noise).item(), torch.max(pred_noise).item())
        print('v1 sampled: ', torch.min(data).item(), torch.max(data).item())
        wandb.log({"v1_sample_min": round(torch.min(data).item(), 5)})  
        wandb.log({"v1_sample_max": round(torch.max(data).item(), 5)})  

        return data
    
    @torch.no_grad()
    def ddim_sample_fromv2(self, init_data, label, age, pos, shape, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps_fromv2, self.num_timesteps_fromv2, self.ddim_sampling_eta,

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        data = init_data
        # print('init_data: ', data)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            pred_noise, x_start, *_ = self.model_predictions(data, label, age, pos, time_cond, self_cond, clip_x_start = clip_denoised)
            data += pred_noise 

        print('v2+@ last noise min max: ', torch.min(pred_noise).item(), torch.max(pred_noise).item())
        wandb.log({"v2+@_sample_min": round(torch.min(data).item(), 5)})  
        wandb.log({"v2+@_sample_max": round(torch.max(data).item(), 5)})  
        return data
    
    @torch.no_grad()
    def sample(self, label, age, batch_size = 16):
        num_node, channels = self.num_node, self.channels
        num_visit = label.shape[0]

        sampled_result = []
        for v in range(num_visit):
            if v == 0:
                sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample_init
                init_data = torch.randn((batch_size, channels, num_node)).cuda()
                pos = torch.tensor([0]).cuda()
            else:
                sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample_fromv2
                init_data = sampled_result[v-1]
                pos = torch.tensor([1]).cuda()

            sampled = sample_fn(init_data, label[v], age[v], pos, (batch_size, channels, num_node))
            sampled_result.append(sampled[0, :, :].unsqueeze(0)) # sampled with condition

        sampled_result = torch.stack(sampled_result).squeeze()
        return sampled_result

    @autocast(enabled = False)
    def q_sample_default(self, x_start, t, noise=None):
        '''return x_0 + noise '''
        return (
        extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_sample_between_visits(self, label, age, t):
        """x_start is one of the 2nd ~ the last visits of a subject"""
        label_prev, label_curr = label[:, 0, :], label[:, 1, :]
        age_prev, age_curr = age[:, 0], age[:, 1]

        label_t = torch.round(label_prev + (label_prev - label_curr) * (t / self.num_timesteps))
        age_t = age_prev + (age_curr - age_prev) * (t / self.num_timesteps)  

        """sample x~P(X|Y=y, A=a)"""
        sampled_x = self.OR_model.sample_from_cdf(label_t, age_t)

        return sampled_x

    def p_losses(self, x_start, label, age, visit, t, x_diff = None, noise = None): 
    
        if visit == 0: # first visit
            b, c, n = x_start.shape # batch, channel, num_node
            noise = default(noise, lambda: torch.randn(b, c, n).cuda())
            x = self.q_sample_default(x_start, t, noise) # x_t = x_start + noise

            x_self_cond = None
            pos = torch.tensor([[0]]).cuda()
            model_out = self.model(x, t, label, age, pos, x_self_cond) 

        else:
            eps = 0.001
            x_t = self.q_sample_between_visits(label, age, t) # x_t is sampled from P(X|Y=y, A=a)
            x_t = torch.clamp(x_t, min=x_start[:, :, 0, :] + x_diff * t - eps, max = x_start[:, :, 0, :] + x_diff * t + eps)

            x_t_plus_1 = self.q_sample_between_visits(label, age, t+1) # x_(t+1) is sampled from P(X|Y=y, A=a)
            x_t_plus_1 = torch.clamp(x_t_plus_1, min=x_start[:, :, 0, :] + x_diff * (t+1) - eps, max = x_start[:, :, 0, :] + x_diff * (t+1) + eps)

            b, v, c = label.shape # batch, 2, class

            x_self_cond = None
            pos = torch.tensor([[1]]).cuda()
            model_out = self.model(x_t, t, label[:, -1, :], age[:, -1], pos, x_self_cond) 
            noise = x_t_plus_1 - x_t

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target)
        return loss
    

    def forward(self, data, label, age, *args, **kwargs):
        b, c, v, n, device, num_node, = *data.shape, data.device, self.num_node
        assert n == num_node, f'seq length must be {num_node}'
        
        loss_per_visit = []
        for i in range(v):
            if i == 0: # first visit
                t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
                p_loss = self.p_losses(data[:, :, i, :], label[:, i, :], age[:, i], i, t, *args, **kwargs)

            else: # 2nd ~ last visits
                t = torch.randint(0, self.num_timesteps_fromv2, (b,), device=device).long()
                x_diff = (data[:, :, i, :] - data[:, :, i-1, :])/self.num_timesteps_fromv2

                p_loss += self.p_losses(data[:, :, i-1:i+1, :], label[:, i-1:i+1, :], age[:, i-1:i+1], i, t, x_diff, *args, **kwargs)
            loss_per_visit.append(p_loss.item())
        return p_loss, loss_per_visit

# trainer class

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        warmup_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1,
        num_samples = 1, # 4
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        optim = 'SGD',
        norm_min = 0.5,
        norm_max = 4.42,
        warmup = False,
        alpha = 10
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.alpha = alpha

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps
        self.warmup_num_steps = warmup_num_steps
        self.optim = optim
        self.warmup = warmup

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        if self.optim == 'SGD':
            self.opt = SGD(diffusion_model.parameters(), lr = train_lr)
        elif self.optim == 'Adam':
            self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = diffusion_model 
            self.ema.to(self.device)

        self.model.OR_model.get_device(self.device) ### put OR_model onto designated GPUs

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, pth):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }
        torch.save(data, pth)

    def load(self, pth):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(pth, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self, test_data, test_label, test_age):
        accelerator = self.accelerator
        mse = nn.MSELoss()

        loss_min, mse_min = 50000000, 50000000
        self.step = 0
        test_length = test_age.shape[0]
        self.scheduler = LambdaLR(optimizer=self.opt, lr_lambda=lambda epoch: 0.99 ** self.step)
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.
                v1_loss, v2_loss = 0., 0.

                with torch.autograd.detect_anomaly():
                    for _ in range(self.gradient_accumulate_every):
                        data = next(self.dl) 
                        x, label, age = data[0], data[1], data[2]

                        with self.accelerator.autocast():
                            loss, loss_per_visit = self.model(x, label, age)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                            v1_loss += loss_per_visit[0] / self.gradient_accumulate_every
                            v2_loss += loss_per_visit[1] / self.gradient_accumulate_every

                        self.accelerator.backward(loss)
                pbar.set_description(f'loss: {total_loss:.4f}')
                wandb.log({"train_loss": round(total_loss, 5),
                           "loss_v1": round(v1_loss, 5),
                           "loss_v2": round(v2_loss, 5)
                        })  

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                self.scheduler.step()
                if accelerator.is_main_process:
                    # self.ema.update()
                    self.ema.train()

                    if (self.step != 0 and self.step % self.save_and_sample_every == 0) or (self.step == self.train_num_steps - 1):
                        # self.ema.ema_model.eval()
                        self.ema.eval()
                        print('Sampling in Epoch #', self.step)

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(map(lambda n: self.ema.sample(label=test_label, age=test_age, batch_size=n), batches))

                        all_samples = torch.cat(all_samples_list, dim = 0)

                        sampled_seq_norm = (all_samples - self.norm_min)/(self.norm_max - self.norm_min) 
                        mse_all = 0
                        mse_v1, mse_rest = 0, 0
                        for idx in range(self.num_samples):
                            # calculate mse with gt and generated samples 
                            mse_all += mse(all_samples[idx*test_length : (idx+1)*test_length, :], test_data)
                            mse_v1 += mse(all_samples[idx*test_length, :], test_data[0])
                            mse_rest += mse(all_samples[idx*(test_length)+1 : (idx+1)*test_length, :], test_data[1:])

                        mse_mean = (mse_v1 + mse_rest) / self.num_samples
                        
                        for i in range(test_length):
                            print('MSE v' + str(i) + ': ', mse(all_samples[i], test_data[i]).item())
                        print('MSE of the target: ', mse_mean.item())

                        wandb.log({"mse": round(mse_mean.item(), 5)})  
                        wandb.log({"mse v1": round(mse_v1.item(), 5)})  
                        wandb.log({"mse v2@": round(mse_rest.item(), 5)})  
                        
                        best_flag = False
                        if mse_mean < mse_min:
                            mse_min = mse_mean.item()
                            best_flag = True

                        for idx in range(self.num_samples):    
                            # visualize normalized samples
                            image = wandb.Image(sampled_seq_norm[idx*test_length :(idx+1)*test_length, :], caption=f'sample-{milestone}-{idx}.png', file_type="jpg")
                            if best_flag:
                                wandb.run.log({"train sample_" + str(milestone) + '_' + str(idx) + '_mse_' + str(round(mse_min, 3)): [image]})
                                torch.save(all_samples, os.path.join(self.results_folder, 'sample-' + str(milestone) + '_' + str(idx) + '_mse_' + str(round(mse_min, 3))))
                            else:
                                wandb.run.log({"train sample_" + str(milestone) + '_' + str(idx): [image]})
                                torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                
                if total_loss < loss_min:
                    loss_min = total_loss
                    pth = os.path.join(self.results_folder, 'model_' + str(self.step) + '_' + str(round(loss_min, 3)) + '_' + str(round(mse_min, 3)) + '.pt')
                    self.save(pth)
                pbar.update(1)

        accelerator.print('training complete')