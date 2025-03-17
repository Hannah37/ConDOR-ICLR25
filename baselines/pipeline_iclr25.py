import tomli
import shutil
import os, time
import argparse
from train import train
import pandas as pd
import matplotlib.pyplot as plt
import zero
import lib
from lib import round_columns
import torch
from scripts.utils_train import get_model, make_dataset
import numpy as np
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from tab_ddpm.utils import ohe_to_categories
SPLIT = 'val'

def load_config(path) :
    with open(path, 'rb') as f:
        return tomli.load(f)
    
def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', default='exp/Amyloid/tab_ddpm1/config.toml')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true',  default=True)
    # parser.add_argument('--eval', action='store_true',  default=True)
    parser.add_argument('--change_val', action='store_true',  default=False)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    if 'device' in raw_config:
        device = torch.device(raw_config['device'])
    else:
        device = torch.device('cuda:0')

    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)

    if args.train:
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=args.change_val
        )
    if args.sample:
        start = time.time()
        sample(
            num_samples=raw_config['sample']['num_samples'],
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=args.change_val
        )
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Required Time {:0>2}h {:0>2}m {:05.2f}s".format(int(hours), int(minutes), seconds))

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))


def sample(
    parent_dir,
    real_data_path = 'data/higgs-small',
    batch_size = 2000,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False
):
    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()
    trainable_params = sum(p.numel() for p in diffusion.parameters())
    print("trainable_params", trainable_params)
    exit()

    x_gen = sample_test(diffusion, torch.from_numpy(D.y[SPLIT]), torch.from_numpy(D.X_cat[SPLIT]))
    # x_gen = sample_test(diffusion, torch.from_numpy(D.y['val']))

    X_gen = x_gen.numpy()

    num_numerical_features = num_numerical_features + int(D.is_regression and not model_params["is_y_cond"])

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        # np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict['cat_encoding'] == 'one-hot':
            X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

    if num_numerical_features_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        # np.save(os.path.join(parent_dir, f'X_num_unnorm_{SPLIT}'), X_gen[:, :num_numerical_features])
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]

        X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        # if model_params['num_classes'] == 0:
        #     y_gen = X_num[:, 0]
        #     X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)
    
    # if num_numerical_features != 0:
    #     print("Num shape: ", X_num.shape)
    #     np.save(os.path.join(parent_dir, f'X_num_norm_{SPLIT}'), X_num)
    # if num_numerical_features < X_gen.shape[1]:
    #     np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
    # np.save(os.path.join(parent_dir, 'y_train'), y_gen)

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

@torch.no_grad()
def sample_test(self: GaussianMultinomialDiffusion, y, x_cat):
    bsize = 500
    b = y.shape[0] * bsize
    device = self.log_alpha.device
    z_norm = torch.randn((b, self.num_numerical_features), device=device)

    has_cat = self.num_classes[0] != 0
    log_z = torch.zeros((b, 0), device=device).float()
    if has_cat:
        uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
        log_z = self.log_sample_categorical(uniform_logits)

    sample_gen = torch.zeros(y.shape[0], z_norm.shape[1]).cpu()
    remain_sample_id = torch.arange(len(sample_gen))
    maxiter = 10
    itern = 0
    print(f"Sampling {len(remain_sample_id)} samples")
    while len(remain_sample_id) > 0 and itern <= maxiter:
        itern += 1
        b = len(remain_sample_id) * bsize
        out_dict = {'y': torch.cat([y[remain_sample_id].long().to(device) for _ in range(bsize)])}
        batch_x_cat = torch.cat([x_cat[remain_sample_id].to(device) for _ in range(bsize)])

        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits)

        for i in reversed(range(0, self.num_timesteps)):
            print(f'Remain {len(remain_sample_id)} Samples, timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                **out_dict
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['sample']
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t, out_dict)

        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        
        batch_mask_nan = torch.any(sample.isnan(), dim=1)
        batch_mask_age_cond = torch.any(batch_x_cat != z_cat, dim=-1).cpu()
        batch_mask_nan = torch.logical_or(batch_mask_nan, batch_mask_age_cond)
        batch_mask_nan = batch_mask_nan.reshape(bsize, len(remain_sample_id)) # B x N
        mask_nan = batch_mask_nan.all(dim=0) # N
        mi = torch.where(~mask_nan)[0]
        bi_ls, bmi_ls = torch.where(~batch_mask_nan)
        sample_ind = []
        for bi, bmi in zip(bi_ls, bmi_ls):
            if bmi in mi:
                sample_ind.append(bi*len(remain_sample_id) + bmi)
                mi = mi[mi!=bmi]
        
        sample_ind = torch.LongTensor(sample_ind)
        sample_gen[remain_sample_id[~mask_nan]] = z_norm.cpu()[sample_ind].float()
        remain_sample_id = remain_sample_id[mask_nan]

    return sample_gen

if __name__ == '__main__':
    main()