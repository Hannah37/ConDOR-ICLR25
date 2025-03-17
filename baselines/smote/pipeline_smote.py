import tomli
import shutil
import os, torch, time
import argparse
from sample_smote import *
# from scripts.eval_catboost import train_catboost
# from scripts.eval_mlp import train_mlp
# import zero
# import lib
import lib.util as util
import lib.data as data
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

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
    parser.add_argument('--config', metavar='FILE',  default='exp/Tau/smote1/config.toml')
    parser.add_argument('--sample', action='store_true',  default=True)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)

    args = parser.parse_args()
    raw_config = load_config(args.config)
    np.random.seed(raw_config['seed'])
    raw_config['seed'] = np.random.randint(10000)
    # timer = zero.Timer()
    # timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)
    if args.sample:
        # sample_given_cond(
        #     parent_dir=raw_config['parent_dir'],
        #     real_data_path=raw_config['real_data_path'],
        #     **raw_config['smote_params'],
        #     seed=raw_config['seed'],
        #     change_val=args.change_val,
        #     split='train'
        # )
        start = time.time()
        sample_given_cond(
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            **raw_config['smote_params'],
            seed=raw_config['seed'],
            change_val=args.change_val,
            split='val'
        )
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Required Time {:0>2}h {:0>2}m {:05.2f}s".format(int(hours), int(minutes), seconds))

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))
    if args.eval:
        if raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )
        # elif raw_config['eval']['type']['eval_model'] == 'mlp':
        #     train_mlp(
        #         parent_dir=raw_config['parent_dir'],
        #         real_data_path=raw_config['real_data_path'],
        #         eval_type=raw_config['eval']['type']['eval_type'],
        #         T_dict=raw_config['eval']['T'],
        #         seed=raw_config['seed'],
        #         change_val=args.change_val
        #     )

    # print(f'Elapsed time: {str(timer)}')

def sample_given_cond(
    parent_dir,
    real_data_path,
    eval_type = "synthetic",
    k_neighbours = 5,
    frac_samples = 1.0,
    frac_lam_del = 0.0,
    change_val = False,
    save = True,
    seed = 0, split='val'
):
    max_iter = 10
    device = 'cuda:0'
    lam1 = 0.0 + frac_lam_del / 2
    lam2 = 1.0 - frac_lam_del / 2

    real_data_path = Path(real_data_path)
    info = util.load_json(real_data_path / 'info.json')
    is_regression = info['task_type'] == 'regression'

    X_num = {}
    X_cat = {}
    y_dict = {}

    if change_val:
        X_num['train'], X_cat['train'], y_dict['train'], X_num['val'], X_cat['val'], y_dict['val'] = data.read_changed_val(real_data_path)
    else:
        X_num['train'], X_cat['train'], y_dict['train'] = data.read_pure_data(real_data_path, 'train')
        X_num['val'], X_cat['val'], y_dict['val'] = data.read_pure_data(real_data_path, 'val')
    X_num['test'], X_cat['test'], y_dict['test'] = data.read_pure_data(real_data_path, 'test')


    X = {k: X_num[k] for k in X_num.keys()}

    if is_regression:
        X['train'] = np.concatenate([X["train"], y_dict["train"].reshape(-1, 1)], axis=1, dtype=object)
        y_dict['train'] = np.where(y_dict["train"] > np.median(y_dict["train"]), 1, 0)
    
    n_num_features = X['train'].shape[1]
    n_cat_features = X_cat['train'].shape[1] if X_cat['train'] is not None else 0
    cat_features = list(range(n_num_features, n_num_features+n_cat_features))
    # print(cat_features)

    scaler = MinMaxScaler().fit(X["train"])
    X["train"] = scaler.transform(X["train"]).astype(object)

    if X_cat['train'] is not None:
        for k in X_num.keys():
            X[k] = np.concatenate([X[k], X_cat[k]], axis=1, dtype=object)

    # print("Before:", X['train'].shape)
    
    sample_gen = torch.zeros(y_dict[split].shape[0], X_num[split].shape[1]).cpu()
    remain_sample_id = torch.arange(len(sample_gen))
    org_X_cat = torch.from_numpy(X_cat[split].astype(int)).to(device)
    org_y = torch.from_numpy(y_dict[split]).to(device)
    save_X_num = X_num[split]
    sample_iter = 0
    while len(remain_sample_id) > 0:
        if sample_iter > max_iter: break
        sample_iter += 1
        print(f'Remain {len(remain_sample_id)} Samples being generating')
        X_cat = org_X_cat[remain_sample_id.to(device)]
        y = org_y[remain_sample_id.to(device)]
        
        gen_X_num, gen_X_cat, gen_y = sample_once(
            X,
            y_dict,
            X_num,
            scaler,
            cat_features,
            eval_type,
            k_neighbours,
            frac_samples,
            n_cat_features,
            lam1,
            lam2,
            is_regression,
            seed
        )
        save_X_num = gen_X_num[:, :save_X_num.shape[1]]

        save_X_num = torch.from_numpy(save_X_num.astype(float))
        gen_X_cat = torch.from_numpy(gen_X_cat.astype(int)).to(device)
        gen_y = torch.from_numpy(gen_y.astype(int)).to(device)
        batch_mask_cat_cond = torch.any(X_cat[:, None] != gen_X_cat[None, :], dim=-1).cpu() # N_y x N_sample
        batch_mask_y_cond = (y[:, None] != gen_y[None, :]).cpu() # N_y x N_sample
        batch_mask_cond = torch.logical_or(batch_mask_y_cond, batch_mask_cat_cond)
        mask_cond = batch_mask_cond.all(dim=1) # N_y
        all_yi = torch.where(~mask_cond)[0]
        yi_ls, samplei_ls = torch.where(~batch_mask_cond)
        sample_ind = []
        for yi, samplei in zip(yi_ls, samplei_ls):
            if yi in all_yi:
                sample_ind.append(samplei)
                all_yi = all_yi[all_yi!=yi]
        
        sample_ind = torch.LongTensor(sample_ind)
        sample_gen[remain_sample_id[~mask_cond]] = save_X_num[sample_ind].float()
        remain_sample_id = remain_sample_id[mask_cond]
    print(sample_gen.shape)
    path = Path(parent_dir)
    np.save(path / f'X_num_{split}', sample_gen.numpy())


def sample_once(
    X,
    y,
    X_num,
    scaler,
    cat_features,
    eval_type,
    k_neighbours,
    frac_samples,
    n_cat_features,
    lam1,
    lam2,
    is_regression,
    seed
):
    
    if eval_type != 'real':
        strat = {k: int((1 + frac_samples) * np.sum(y['train'] == k)) for k in np.unique(y['train'])}
        # print(strat)
        if n_cat_features > 0:
            sm = MySMOTENC(
                lam1=lam1,
                lam2=lam2,
                random_state=seed,
                k_neighbors=k_neighbours,
                categorical_features=cat_features,
                sampling_strategy=strat
            )
        else:
            sm = MySMOTE(
                lam1=lam1,
                lam2=lam2,
                random_state=seed,
                k_neighbors=k_neighbours,
                sampling_strategy=strat
            )
        print(vars(sm))
        X_res, y_res = sm.fit_resample(X['train'], y['train'])
        if is_regression:
            X_res[:, :X_num["train"].shape[1]+1] = scaler.inverse_transform(X_res[:, :X_num["train"].shape[1]+1])
            y_res = X_res[:, X_num["train"].shape[1]]
            X_res = np.delete(X_res, [X_num["train"].shape[1]], axis=1)
        else:
            X_res[:, :X_num["train"].shape[1]] = scaler.inverse_transform(X_res[:, :X_num["train"].shape[1]])
            y_res = y_res.astype(int)

        if eval_type == "synthetic":
            X_res = X_res[X['train'].shape[0]:]
            y_res = y_res[X['train'].shape[0]:]
        
    disc_cols = []
    for col in range(X_num["train"].shape[1]):
        uniq_vals = np.unique(X_num["train"][:, col])
        if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
            disc_cols.append(col)
    if len(disc_cols):
        X_res[:, :X_num["train"].shape[1]] = data.round_columns(X_num["train"], X_res[:, :X_num["train"].shape[1]], disc_cols)
    
    
    
    X_num = X_res[:, :-n_cat_features]
    X_cat = X_res[:, -n_cat_features:]
        
    return X_num, X_cat, y_res

from typing import Any, Callable, List, Dict, Type, Optional, Tuple, TypeVar, Union, cast, get_args, get_origin

def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)


_CONFIG_NONE = '__none__'
RawConfig = Dict[str, Any]
def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def pack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x is None, _CONFIG_NONE))
    return config


def load_config(path: Union[Path, str]) -> Any:
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))

if __name__ == '__main__':
    main()