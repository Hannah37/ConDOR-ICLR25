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

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from scipy.stats import wasserstein_distance_nd

from accelerate import Accelerator
# from ema_pytorch import EMA
import copy

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiplicativeLR

from version import __version__

import os
import sys
import wandb
import pickle
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../'))
import or_batch

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
class warmup_Dataset1D(Dataset):
    def __init__(self, args, status='train'):
        super().__init__()
        self.dir = args.dir
        self.classes = args.classes
        self.status = status # either 'train' or 'test'

        self.min_age = args.age_min
        self.max_age = args.age_max

        self.data = []
        self.label = []
        self.age = []

        self.load_data()

    def load_data(self):
        self.max_visits = 1
        for filename in os.listdir(os.path.join(self.dir, self.status+'_data')):
            data = torch.load(os.path.join(self.dir, self.status+'_data', filename))
            data = data.unsqueeze(0).type(torch.float32) 
            
            n_visit = data.shape[1]
            if n_visit == 1:
                print(filename, 'The number of visit is 1')
                break
            for v in range(n_visit):
                self.data.append(data[:, v, :])

            label = torch.load(os.path.join(self.dir, self.status+'_label', filename[:-3] + '.pt'))
            '''one-hot'''
            subj_label = F.one_hot(label, num_classes=self.classes).type(torch.float32)

            for v in range(n_visit):
                self.label.append(subj_label[v, :]) 

            age = torch.load(os.path.join(self.dir, self.status+'_age', filename[:-3]) + '.pt')
            age_ = [(a - self.min_age)/(self.max_age - self.min_age) for a in age]

            if (len(label) != len(age)) or (len(label) != data.shape[1]):
                print(filename, data.shape, len(label), len(age))
                print("Length of label, age, data should be same")

            assert (len(label) == len(age)) and (data.shape[1] == len(label)), "Length of label, age, data should be same"

            age_ = torch.tensor(age_).type(torch.float32)
            for v in range(n_visit):
                self.age.append(age_[v])

    def collate_fn(self, samples):
        collate_x, collate_y, collate_a, collate_mask = [], [], [], []
        for sample in samples:
            x, y, a = sample[0], sample[1], sample[2]
            n_visit = x.shape[1]
            mask = torch.ones(n_visit)
            collate_x.append(x)
            collate_y.append(y)
            collate_a.append(a)
            collate_mask.append(mask)
        X, Y, A, M = torch.stack(collate_x), torch.stack(collate_y), torch.stack(collate_a), torch.stack(collate_mask)
        return X, Y, A, M

    def max_visit(self):
        return self.max_visits # 6 for ADNI
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx] 
        label = self.label[idx]
        age = self.age[idx] 
        return data, label, age
    
# data
class Dataset1D(Dataset):
    def __init__(self, args, status='train'):
        super().__init__()
        self.dir = args.dir
        self.classes = args.classes
        self.status = status # either 'train' or 'test'

        self.min_age = args.age_min
        self.max_age = args.age_max

        self.data = []
        self.label = []
        self.age = []

        self.load_data()

    def load_data(self):
        self.max_visits = 0

        for filename in os.listdir(os.path.join(self.dir, self.status+'_data')):
            data = torch.load(os.path.join(self.dir, self.status+'_data', filename))
            data = data.unsqueeze(0).type(torch.float32) # num_channel = 1

            if data.shape[1] == 1:
                print(filename, 'The number of visit is 1')
                break
            self.data.append(data)

            label = torch.load(os.path.join(self.dir, self.status+'_label', filename[:-3] + '.pt'))
            '''one-hot'''
            subj_label = F.one_hot(label, num_classes=self.classes).type(torch.float32)
            self.label.append(subj_label) 

            if len(label) > self.max_visits: self.max_visits = len(label)

            age = torch.load(os.path.join(self.dir, self.status+'_age', filename[:-3]) + '.pt')
            age_ = [(a - self.min_age)/(self.max_age - self.min_age) for a in age]

            if (len(label) != len(age)) or (len(label) != data.shape[1]):
                print(filename, data.shape, len(label), len(age))
                print("Length of label, age, data should be same")

            assert (len(label) == len(age)) and (data.shape[1] == len(label)), "Length of label, age, data should be same"

            age_ = torch.tensor(age_).type(torch.float32)
            self.age.append(age_)

    def collate_fn(self, samples):
        collate_x, collate_y, collate_a, collate_mask = [], [], [], []
        
        for sample in samples:
            x, y, a = sample[0], sample[1], sample[2]
            n_visit = x.shape[1]
            diff = self.max_visits - n_visit
            mask = torch.ones(n_visit)
            if diff > 0:
                zero_pad_x = torch.zeros(size=(1, diff, 148))
                zero_pad_y = torch.zeros(size=(diff, self.classes))
                zero_pad_a = torch.zeros(size=(diff,))
                zero_pad_mask = torch.zeros(size=(diff,))

                collate_x.append(torch.cat([x, zero_pad_x], dim=1)) 
                collate_y.append(torch.cat([y, zero_pad_y], dim=0))
                collate_a.append(torch.cat([a, zero_pad_a], dim=0))
                collate_mask.append(torch.cat([mask, zero_pad_mask], dim=0))
            else:
                collate_x.append(x)
                collate_y.append(y)
                collate_a.append(a)
                collate_mask.append(mask)

        X, Y, A = torch.stack(collate_x), torch.stack(collate_y), torch.stack(collate_a)
        M = torch.stack(collate_mask)

        return X, Y, A, M
    
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
        if x.size() == torch.Size([]):
            x = x.unsqueeze(0)
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

    def forward(self, x): 
        x = x.unsqueeze(-1) 
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

    def forward(self, x): 
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

    def forward(self, x): 
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
        self.cond_mix1 = EmbedFC_cond_mix(2*cond_in, 2*cond_out)
        self.cond_mix2 = EmbedFC_cond_mix(cond_in, cond_out)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in + 2*cond_out, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out + dim_in + cond_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(ind, dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding = 1) 
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, label, age, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        '''embedding of conditions'''
        age_emb1 = self.age_embed1(age) 
        age_emb2 = self.age_embed2(age) 
        label_emb1 = self.label_embed1(label.type(torch.float32))
        label_emb2 = self.label_embed2(label.type(torch.float32)) 
      
        '''concat conditions'''
        cemb1 = torch.cat((age_emb1, label_emb1), dim=-1) 
        cemb2 = torch.cat((age_emb2, label_emb2), dim=-1) 

        cemb1 = self.cond_mix1(cemb1) 
        cemb2 = self.cond_mix2(cemb2) 
        x = self.init_conv(x)
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
        self.OR_model = or_batch.OrdinalRegression(self.args)
    
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
        self.num_timesteps_fromv2 = 100 # 10
        wandb.config.T_fromv2 = self.num_timesteps_fromv2

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

    def model_predictions(self, x, label, age, t, get_warmup = False, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        if get_warmup == True:
            model_output = self.warmup_model.model(x, t, label, age, x_self_cond) ## 
        else:
            model_output = self.model(x, t, label, age, x_self_cond)
 
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise) ########## 겁나 큼

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

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            data, x_start = self.p_sample(data, label, age, t, self_cond)

        return data

    @torch.no_grad()
    def ddim_sample_init(self, init_data, label, age, get_warmup, shape, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        data = init_data
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            pred_noise, x_start, *_ = self.model_predictions(data, label, age, time_cond, get_warmup, self_cond, clip_x_start = clip_denoised)

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

        wandb.log({"v1_sample_min": round(torch.min(data).item(), 5)})  
        wandb.log({"v1_sample_max": round(torch.max(data).item(), 5)})  

        return data
    
    @torch.no_grad()
    def ddim_sample_fromv2(self, init_data, label, age, shape, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps_fromv2, self.num_timesteps_fromv2, self.ddim_sampling_eta,

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        data = init_data

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            pred_noise, x_start, *_ = self.model_predictions(data, label, age, time_cond, self_cond, clip_x_start = clip_denoised)
            data += pred_noise 

        print('v2+@ last noise min max: ', torch.min(pred_noise).item(), torch.max(pred_noise).item())
        wandb.log({"v2+@_sample_min": round(torch.min(data).item(), 5)})  
        wandb.log({"v2+@_sample_max": round(torch.max(data).item(), 5)})  
        return data
    
    @torch.no_grad()
    def sample(self, label, age, get_warmup=False, batch_size = 65):
        num_node, channels = self.num_node, self.channels
        sampled_result = []
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample_init
        init_data = torch.randn((batch_size, channels, num_node)).cuda()
    
        sampled = sample_fn(init_data, label, age, get_warmup, (batch_size, channels, num_node))
        sampled_result.append(sampled) 

        sampled_result = torch.stack(sampled_result).squeeze()
        return sampled_result  
      
    @torch.no_grad()
    def sample_fromv2(self, label, age, batch_size = 16):
        num_node, channels = self.num_node, self.channels
        num_visit = label.shape[1]

        sampled_result_v0, sampled_result_from2nd = [], []

        sampled = self.sample(label[:, 0, :], age[:, 0], True, batch_size)
        init_sample = sampled.squeeze().unsqueeze(1)

        sampled_result_v0.append(sampled.squeeze().clone())

        for v in range(1, num_visit):
            sample_fn = self.ddim_sample_fromv2
            if v == 1:
                prev_data = init_sample
            else:
                prev_data = sampled.squeeze().unsqueeze(1)
                
            sampled = sample_fn(prev_data, label[:, v, :], age[:, v], (batch_size, channels, num_node))
            sampled = sampled.squeeze()
            sampled_result_from2nd.append(sampled.clone()) 
        sampled_result_v0 = torch.stack(sampled_result_v0)
        sampled_result_from2nd = torch.stack(sampled_result_from2nd)

        sampled_result = torch.cat((sampled_result_v0, sampled_result_from2nd), 0)
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
        n_classes = label.shape[-1]
        label_prev, label_curr = label[:, 0, :], label[:, 1, :]
        label_prev = torch.argmax(label_prev, dim=1)
        label_curr = torch.argmax(label_curr, dim=1)
        age_prev, age_curr = age[:, 0], age[:, 1]

        label_t = label_prev + (label_curr - label_prev) * (t.squeeze() / self.num_timesteps_fromv2)
        label_t = torch.clamp(label_t, min=0)
        label_t = torch.round(label_t).to(torch.int64)

        label_t = F.one_hot(label_t, num_classes=n_classes)

        age_t = age_prev + (age_curr - age_prev) * (t.squeeze() / self.num_timesteps_fromv2) 

        """sample x~P(X|Y=y, A=a)"""
        sampled_x = self.OR_model.sample_from_cdf(label_t, age_t) 

        return sampled_x

    def get_warmup_model(self, warmup_model):
        self.warmup_model = warmup_model

    def p_losses(self, x_start, label, age, visit, t, x_diff=None, noise=None): 
        if visit == 0: # first visit
            b, c, n = x_start.shape # batch, channel, num_node
            noise = default(noise, lambda: torch.randn(b, c, n).cuda())
            x = self.q_sample_default(x_start, t, noise) # x_t = x_start + noise

            x_self_cond = None
            model_out = self.model(x, t, label, age, x_self_cond) 
            return model_out, noise
        
        else: # 2nd ~ last visit
            b, c, _, n = x_start.shape
            t = t.unsqueeze(1).unsqueeze(2)

            cohort_x_t = self.q_sample_between_visits(label, age, t) # x_t is sampled from P(X|Y=y, A=a)
            cohort_x_t_plus_1 = self.q_sample_between_visits(label, age, t+1) # x_(t+1) is sampled from P(X|Y=y, A=a)
            cohort_noise = cohort_x_t_plus_1 - cohort_x_t
        
            x_t = x_start[:, :, 0, :] + x_diff * t
            x_t_plus_1 = x_start[:, :, 0, :] + x_diff * (t+1)

            b, v, c = label.shape 
            x_self_cond = None
            t = t.squeeze()
            model_out = self.model(x_t, t, label[:, -1, :], age[:, -1], x_self_cond) 
            subject_noise = x_t_plus_1 - x_t

            noise = cohort_noise * self.alpha + subject_noise * (1-self.alpha)
            return model_out, noise

    def get_alpha(self, alpha):
        self.alpha = alpha
    
    def forward(self, data, label, age, *args, **kwargs):
        b, c, v, n, device, num_node, = *data.shape, data.device, self.num_node
        assert n == num_node, f'seq length must be {num_node}'

        if v == 1:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            model_out, noise = self.p_losses(data[:, :, 0, :], label[:, 0, :], age[:, 0], 0, t, *args, **kwargs)
            return model_out, noise
        else: 
            model_out_all, noise_all = [], []

            for i in range(1, v):
                t = torch.randint(0, self.num_timesteps_fromv2, (b,), device=device).long()
                x_diff = (data[:, :, i, :] - data[:, :, i-1, :])/self.num_timesteps_fromv2
                model_out, noise = self.p_losses(data[:, :, i-1:i+1, :], label[:, i-1:i+1, :], age[:, i-1:i+1], i+1, t, x_diff, *args, **kwargs)
                model_out_all.append(model_out)
                noise_all.append(noise)
            model_out_all = torch.stack(model_out_all, dim=1).squeeze()
            noise_all = torch.stack(noise_all, dim=1).squeeze()
        return model_out_all, noise_all

# trainer class

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        train_batch_size = 16,
        test_size = 65,
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
        alpha = 10,
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
        self.model.get_alpha(self.alpha)

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
        dl = DataLoader(train_dataset, batch_size = train_batch_size, collate_fn=train_dataset.collate_fn, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        test_dl = DataLoader(test_dataset, batch_size=test_size, collate_fn=test_dataset.collate_fn, shuffle = False, pin_memory = True, num_workers = cpu_count())
        self.test_dl = cycle(test_dl)
        self.test_size = test_size

        # optimizer
        self.train_lr = train_lr
        if self.optim == 'SGD':
            self.opt = SGD(diffusion_model.parameters(), lr = self.train_lr)
        elif self.optim == 'Adam':
            self.opt = Adam(diffusion_model.parameters(), lr = self.train_lr, betas = adam_betas)

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
        model.load_state_dict(data['model'], strict=False)

        self.step = 0 
        self.opt.load_state_dict(data['opt'])

        for g in self.opt.param_groups:
            g['lr'] = self.train_lr
        self.scheduler = MultiplicativeLR(optimizer=self.opt, lr_lambda=lambda epoch: 0.99999)

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"], strict=False)

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def kld(self, p, q):
        p = p + 1e-10
        return (p * (p.log() - q.log())).sum(dim=1)
    
    def metrics(self, all_samples, test_x):
        '''
        all_samples: sampled data
        test_x: test data (i.e., brain measurments)

        The input shapes of all_samples and test_x are 2D and both are same: batch x (num_visits x num_node)   
        '''

        if (torch.min(all_samples) < -1e+10) or (torch.max(all_samples) > 1e+10):
            print('sampled values are exploded !!! reduce lr or increase sampling_epoch')
            wd, jsd_mean, rmse = None, None, None
        else:

            # (1) Wasserstein Distance 
            sampled_data_np = all_samples.cpu().numpy()
            real_data_np = test_x.cpu().numpy()

            wd = wasserstein_distance_nd(sampled_data_np, real_data_np)
            print(f"\nWD: {wd:.3f}")

            # (2) Jensen-Shannon divergence
            sampled_data = F.softmax(all_samples, dim=1)
            real_data = F.softmax(test_x, dim=1)

            m = 0.5 * (sampled_data + real_data)

            assert not torch.isnan(self.kld(sampled_data, m)).any().item(), "self.kld(sampled_data, m) has nan"

            jsd = 0.5 * (self.kld(sampled_data, m) + self.kld(real_data, m))
            jsd_mean = jsd.mean().item()
            
            print(f"JSD: {jsd_mean:.3f}")

            # (3) RMSE
            mse = nn.MSELoss()
            mse = mse(all_samples, test_x)
            rmse = torch.sqrt(mse).item()
            print(f"RMSE: {rmse:.3f}")

        return wd, jsd_mean, rmse

    def run_warmup(self, accelerator):
        """
        train & test for all visits
        """
        loss_min = 50000000
        rmse_min, rmse_wd_min, rmse_jsd_min = 50000000, 50000000, 50000000
        wd_min, wd_rmse_min, wd_jsd_min = 50000000, 50000000, 50000000
        jsd_min, jsd_wd_min, jsd_rmse_min = 50000000, 50000000, 50000000

        print('warmup started!!')
        self.scheduler = LambdaLR(optimizer=self.opt, lr_lambda=lambda epoch: 0.99999 ** self.step)
        with tqdm(initial = self.step, total = self.warmup_num_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.warmup_num_steps:
                total_loss = 0.
                with torch.autograd.detect_anomaly():
                    for _ in range(self.gradient_accumulate_every):
                        data = next(self.dl) 
                        x, label, age, mask = data[0], data[1], data[2], data[3]

                        # print(x.shape, label.shape, age.shape) # torch.Size([1, 1, 2, 148]) torch.Size([1, 2, 5]) torch.Size([1, 2])
                        with self.accelerator.autocast():
                            model_out, noise = self.model(x[:, :, :].unsqueeze(-2), label[:, :].unsqueeze(-2), age[:].unsqueeze(-1))
                            loss = F.mse_loss(model_out, noise)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                        self.accelerator.backward(loss)
                pbar.set_description(f'loss: {total_loss:.4f}')
                wandb.log({"warmup_loss": round(total_loss, 5)})  

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()

                curr_lr = self.opt.param_groups[0]['lr']
                wandb.log({"lr": curr_lr})  

                self.step += 1
                self.scheduler.step()
                if accelerator.is_main_process:
                    self.ema.train()

                    if (self.step != 0 and self.step % self.save_and_sample_every == 0) or (self.step == self.train_num_steps - 1):
                        self.ema.eval()
                        print('Sampling in Epoch #', self.step)

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.test_size)                
                            test_data = next(self.test_dl)
                            test_x, test_label, test_age, test_mask = test_data[0].cuda(), test_data[1].cuda(), test_data[2].cuda(), test_data[3].cuda()
                            # ([205, 1, 148]) torch.Size([205, 5]) torch.Size([205]) torch.Size([205, 148])
                            # print(test_x.shape, test_label.shape, test_age.shape, test_mask.shape)
                            
                            if milestone == 1:
                                fn = 'warmup_GT.png'
                                torch.save(test_x, os.path.join(self.results_folder, fn))
                                image = wandb.Image(test_x[0].unsqueeze(0), caption=fn, file_type="png") ## visualize first data ##########
                                wandb.run.log({"test sample": [image]})
                            all_samples_list = list(map(lambda n: self.ema.sample(label=test_label, age=test_age, batch_size=self.test_size), batches))
                        all_samples = torch.cat(all_samples_list, dim = 0)  # torch.Size([205, 148])
                        wd, js, rmse = self.metrics(all_samples, test_x.squeeze())     

                        if wd != None:
                            fn = 'warmup_sample-' + str(milestone) + '_' + str(round(wd, 3)) + '_' + str(round(js, 3)) + '_' + str(round(rmse, 3)) + '.png'
                            torch.save(all_samples, os.path.join(self.results_folder, fn))
                            image = wandb.Image(all_samples[0].unsqueeze(0), caption=fn, file_type="png") ## visualize first data
                            wandb.run.log({fn: [image]})
                            wandb.log({"warm-up WD": round(wd, 5),
                                    "warm-up JSD": round(js, 5),
                                    "warm-up RMSE": round(rmse, 5)
                                    })  

                            if rmse < rmse_min:
                                rmse_min = rmse
                                rmse_jsd_min = js
                                rmse_wd_min = wd
                                fn = 'warmup_RDM-' + str(milestone) + '_' + str(round(rmse_wd_min, 3)) + '_' + str(round(rmse_jsd_min, 3)) + '_' + str(round(rmse_min, 3)) + '.pt'
                                pth = os.path.join(self.results_folder, fn)
                                self.save(pth)
                                best_rmse_results = {
                                    "best_rmse": rmse,
                                    "best_rmse_wd": wd,
                                    "best_rmse_jsd" : js
                                }
                                wandb.config.update(best_rmse_results, allow_val_change=True)
                                fn = 'warmup_best_sample-' + str(milestone) + '_' + str(round(wd, 3)) + '_' + str(round(js, 3)) + '_' + str(round(rmse, 3)) + '.pt'
                                torch.save(all_samples, os.path.join(self.results_folder, fn))
                            
                            if js < jsd_min:
                                jsd_min = js
                                jsd_rmse_min = rmse
                                jsd_wd_min = wd
                                fn = 'warmup_RDM-' + str(milestone) + '_' + str(round(jsd_wd_min, 3)) + '_' + str(round(jsd_min, 3)) + '_' + str(round(jsd_rmse_min, 3)) + '.pt'
                                pth = os.path.join(self.results_folder, fn)
                                self.save(pth)
                                best_jsd_results = {
                                    "best_jsd_rmse": rmse,
                                    "best_jsd_wd": wd,
                                    "best_jsd" : js
                                }
                                wandb.config.update(best_jsd_results, allow_val_change=True)
                                fn = 'warmup_best_sample-' + str(milestone) + '_' + str(round(wd, 3)) + '_' + str(round(js, 3)) + '_' + str(round(rmse, 3)) + '.pt'
                                torch.save(all_samples, os.path.join(self.results_folder, fn))

                            if wd < wd_min:
                                wd_min = wd
                                wd_jsd_min = js
                                wd_rmse_min = rmse
                                fn = 'warmup_RDM-' + str(milestone) + '_' + str(round(wd_min, 3)) + '_' + str(round(wd_jsd_min, 3)) + '_' + str(round(wd_rmse_min, 3)) + '.pt'
                                pth = os.path.join(self.results_folder, fn)
                                self.save(pth)
                                best_wd_results = {
                                    "best_jsd_rmse": rmse,
                                    "best_jsd_wd": wd,
                                    "best_jsd" : js
                                }
                                wandb.config.update(best_wd_results, allow_val_change=True)
                                fn = 'warmup_best_sample-' + str(milestone) + '_' + str(round(wd, 3)) + '_' + str(round(js, 3)) + '_' + str(round(rmse, 3)) + '.pt'
                                torch.save(all_samples, os.path.join(self.results_folder, fn))
                if total_loss < loss_min:
                    loss_min = total_loss
                pbar.update(1)

        print('Minimum RMSE/JSD/WD: ', rmse_min, rmse_jsd_min, rmse_wd_min)
        print('Minimum JSD/WD/RMSE: ', jsd_min, jsd_rmse_min, jsd_wd_min)
        print('Minimum WD/RMSE/JSD: ', wd_min, wd_jsd_min, wd_rmse_min)
        accelerator.print('Warm-up completed !!!')

    def train(self):
        accelerator = self.accelerator

        if self.warmup == True:
            '''Warm-up for learning initial visit'''
            self.run_warmup(accelerator)
        else:
            pth = os.path.join('warmup_RDM.pt')
            self.load(pth) 
            curr_lr = self.opt.param_groups[0]['lr']
            print('init lr: ', curr_lr)
            warmup_model = copy.deepcopy(self.model)
            self.model.get_warmup_model(warmup_model)

            accelerator.print('RDM loaded !!!')
            accelerator.print('loaded model: ', pth)

            '''Training for all visits'''
            accelerator.print('Training start for all visits !!!')

            loss_min = 50000000
            rmse_min, rmse_wd_min, rmse_jsd_min = 50000000, 50000000, 50000000
            wd_min, wd_rmse_min, wd_jsd_min = 50000000, 50000000, 50000000
            jsd_min, jsd_wd_min, jsd_rmse_min = 50000000, 50000000, 50000000

            with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
                while self.step < self.train_num_steps:
                    total_loss = 0.
                    with torch.autograd.detect_anomaly():
                        for _ in range(self.gradient_accumulate_every):
                            data = next(self.dl) 
                            x, label, age, mask = data[0], data[1], data[2], data[3]

                            with self.accelerator.autocast():
                                model_out, noise = self.model(x, label, age)
                                mask = repeat(mask[:, 1:], 'b v -> b v n', n=self.model.num_node)
                                model_out = model_out * mask
                                noise = noise * mask
                                loss = F.mse_loss(model_out, noise, reduction='sum')
                                loss = loss / mask.sum()
                                loss = loss / self.gradient_accumulate_every
                                total_loss += loss.item()

                            self.accelerator.backward(loss)
                    pbar.set_description(f'loss: {total_loss:.4f}')
                    wandb.log({"train_loss": round(total_loss, 5)})  

                    accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.opt.zero_grad()
                    curr_lr = self.opt.param_groups[0]['lr']
                    wandb.log({"lr": curr_lr})  
                    self.step += 1
                    self.scheduler.step()
                    if accelerator.is_main_process:
                        if (self.step != 0 and self.step % self.save_and_sample_every == 0) or (self.step == self.train_num_steps - 1):
                            self.ema.eval()
                            print('Sampling in Epoch #', self.step)

                            with torch.no_grad():
                                milestone = self.step // self.save_and_sample_every
                                batches = num_to_groups(self.num_samples, self.batch_size)

                                test_data = next(self.test_dl)
                                test_x, test_label, test_age, test_mask = test_data[0].cuda(), test_data[1].cuda(), test_data[2].cuda(), test_data[3].cuda()

                                if milestone == 1:
                                    fn = 'GT.png'
                                    torch.save(test_x, os.path.join(self.results_folder, fn))
                                    image = wandb.Image(test_x[0].unsqueeze(0), caption=fn, file_type="png") ## visualize first data ##########
                                    wandb.run.log({"test sample": [image]})
                                all_samples_list = list(map(lambda n: self.ema.sample_fromv2(label=test_label, age=test_age, batch_size=self.test_size), batches))

                            all_samples = torch.cat(all_samples_list, dim=0)
                            test_mask = repeat(test_mask, 'b v -> b v n', n=self.model.num_node)
                            all_samples = rearrange(all_samples, 'v b n -> b v n')
                            all_samples = all_samples * test_mask
                            all_samples_flat = rearrange(all_samples, 'b v n -> b (v n)')
                            test_x_flat = rearrange(test_x.squeeze(), 'b v n -> b (v n)')
                            wd, js, rmse = self.metrics(all_samples_flat, test_x_flat)    

                            fn = 'sample-' + str(milestone) + '_' + str(round(wd, 3)) + '_' + str(round(js, 3)) + '_' + str(round(rmse, 3)) + '.png'
                            torch.save(all_samples, os.path.join(self.results_folder, fn))
                            image = wandb.Image(all_samples[0].unsqueeze(0), caption=fn, file_type="png") ## visualize first data
                            wandb.run.log({fn: [image]})
                            wandb.log({"WD": round(wd, 5),
                                    "JSD": round(js, 5),
                                    "RMSE": round(rmse, 5)
                                    })  

                            if rmse < rmse_min:
                                rmse_min = rmse
                                rmse_jsd_min = js
                                rmse_wd_min = wd
                                best_rmse_results = {
                                    "best_rmse": rmse,
                                    "best_rmse_wd": wd,
                                    "best_rmse_jsd" : js
                                }
                                wandb.config.update(best_rmse_results, allow_val_change=True)
                                
                                if wd < 10:
                                    fn = 'best_model-' + str(milestone) + '_' + str(round(rmse_wd_min, 3)) + '_' + str(round(rmse_jsd_min, 3)) + '_' + str(round(rmse_min, 3)) + '.pt'
                                    pth = os.path.join(self.results_folder, fn)
                                    self.save(pth)
                                    fn = 'best_sample-' + str(milestone) + '_' + str(round(wd, 3)) + '_' + str(round(js, 3)) + '_' + str(round(rmse, 3)) + '.pt'
                                    torch.save(all_samples, os.path.join(self.results_folder, fn))
                            
                            if js < jsd_min:
                                jsd_min = js
                                jsd_rmse_min = rmse
                                jsd_wd_min = wd
                                best_jsd_results = {
                                    "best_jsd_rmse": rmse,
                                    "best_jsd_wd": wd,
                                    "best_jsd" : js
                                }
                                wandb.config.update(best_jsd_results, allow_val_change=True)
                        
                                if wd < 10:
                                    fn = 'best_model-' + str(milestone) + '_' + str(round(jsd_wd_min, 3)) + '_' + str(round(jsd_min, 3)) + '_' + str(round(jsd_rmse_min, 3)) + '.pt'
                                    pth = os.path.join(self.results_folder, fn)
                                    self.save(pth)
                                    fn = 'best_sample-' + str(milestone) + '_' + str(round(wd, 3)) + '_' + str(round(js, 3)) + '_' + str(round(rmse, 3)) + '.pt'
                                    torch.save(all_samples, os.path.join(self.results_folder, fn))

                            if wd < wd_min:
                                wd_min = wd
                                wd_jsd_min = js
                                wd_rmse_min = rmse
                                best_wd_results = {
                                    "best_jsd_rmse": rmse,
                                    "best_jsd_wd": wd,
                                    "best_jsd" : js
                                }
                                wandb.config.update(best_wd_results, allow_val_change=True)
                                if wd < 10:
                                    fn = 'best_model-' + str(milestone) + '_' + str(round(wd_min, 3)) + '_' + str(round(wd_jsd_min, 3)) + '_' + str(round(wd_rmse_min, 3)) + '.pt'
                                    pth = os.path.join(self.results_folder, fn)
                                    self.save(pth)
                                    fn = 'best_sample-' + str(milestone) + '_' + str(round(wd, 3)) + '_' + str(round(js, 3)) + '_' + str(round(rmse, 3)) + '.pt'
                                    torch.save(all_samples, os.path.join(self.results_folder, fn))

                    if total_loss < loss_min:
                        loss_min = total_loss
                    pbar.update(1)

            print('Minimum RMSE/JSD/WD: ', rmse_min, rmse_jsd_min, rmse_wd_min)
            print('Minimum JSD/WD/RMSE: ', jsd_min, jsd_rmse_min, jsd_wd_min)
            print('Minimum WD/RMSE/JSD: ', wd_min, wd_jsd_min, wd_rmse_min)
            accelerator.print('training complete')