import tomli
import shutil
import os, torch, pickle, time
import argparse
from train_sample_ctabganp import train_ctabgan, sample_ctabgan
from scripts.eval_catboost import train_catboost
import zero
import lib
from model.ctabgan import CTABGAN
from pathlib import Path
import numpy as np

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
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true',  default=False)
    parser.add_argument('--sample', action='store_true',  default=True)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)
    
    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    # raw_config['device'] = 'cuda:1'
    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)
    ctabgan = None
    if args.train:
        ctabgan = train_ctabgan(
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            train_params=raw_config['train_params'],
            change_val=args.change_val,
            device=raw_config['device']
        )
    if args.sample:
        # sample_given_cond(
        #     synthesizer=ctabgan,
        #     parent_dir=raw_config['parent_dir'],
        #     real_data_path=raw_config['real_data_path'],
        #     num_samples=raw_config['sample']['num_samples'],
        #     split='train',
        #     train_params=raw_config['train_params'],
        #     change_val=args.change_val,
        #     seed=raw_config['sample']['seed'],
        #     device=raw_config['device']
        # )
        start = time.time()
        sample_given_cond(
            synthesizer=ctabgan,
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            num_samples=raw_config['sample']['num_samples'],
            split='val',
            train_params=raw_config['train_params'],
            change_val=args.change_val,
            seed=raw_config['sample']['seed'],
            device=raw_config['device']
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

    print(f'Elapsed time: {str(timer)}')

def sample_given_cond(
    synthesizer,
    parent_dir,
    real_data_path,
    num_samples, split='val',
    train_params = {"batch_size": 512},
    change_val=False,
    device="cpu",
    seed=0
):
    max_iter = 10
    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)
    device = torch.device(device)
    print(device)

    X_num, X_cat, y = lib.read_pure_data(real_data_path, split)
    
    X = lib.concat_to_pd(X_num, X_cat, y)

    X.columns = [str(_) for _ in X.columns]

    ctabgan_params = lib.load_json("CTAB-GAN/columns.json")[real_data_path.name]

    cat_features = ctabgan_params["categorical_columns"]
    
    sample_gen = torch.zeros(y.shape[0], X_num.shape[1]).cpu()
    remain_sample_id = torch.arange(len(sample_gen))
    org_X_cat = torch.from_numpy(X_cat.astype(int)).to(device)
    org_y = torch.from_numpy(y).to(device)
    with open(parent_dir / "ctabgan.obj", 'rb')  as f:
        synthesizer = pickle.load(f)
        synthesizer.synthesizer.generator = synthesizer.synthesizer.generator.to(device)
    print(synthesizer.synthesizer.generator)
    trainable_params = sum(p.numel() for p in synthesizer.synthesizer.generator.parameters())
    print("trainable_params", trainable_params)

    sample_iter = 0
    while len(remain_sample_id) > 0:
        if sample_iter > max_iter: break
        sample_iter += 1
        print(f'Remain {len(remain_sample_id)} Samples being generating')
        X_cat = org_X_cat[remain_sample_id.to(device)]
        y = org_y[remain_sample_id.to(device)]
        gen_data = synthesizer.generate_samples(num_samples, seed)

        gen_y = gen_data['y'].values
        if len(np.unique(gen_y)) == 1:
            gen_y[0] = 1

        gen_X_cat = gen_data[cat_features].drop('y', axis=1).values if len(cat_features) else None
        X_num = gen_data.values[:, :X_num.shape[1]] if X_num is not None else None
        
        X_num = torch.from_numpy(X_num.astype(float))
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
        sample_gen[remain_sample_id[~mask_cond]] = X_num[sample_ind].float()
        remain_sample_id = remain_sample_id[mask_cond]

    
    np.save(parent_dir / f'X_num_{split}', sample_gen.numpy())


if __name__ == '__main__':
    main()