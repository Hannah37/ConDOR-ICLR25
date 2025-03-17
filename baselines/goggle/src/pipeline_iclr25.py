
from goggle.GoggleModel import GoggleModel
import pandas as pd
import sys, torch
import numpy as np
import shutil
import os, time

def main():
    device = 'cpu'
    dname = sys.argv[1]
    X_train, X_test = load_data(dname)
    for i in range(3):
        i = i + 1
        gen = GoggleModel(
            ds_name=dname,
            input_dim=X_train.shape[1],
            encoder_dim=64,
            encoder_l=2,
            het_encoding=True,
            decoder_dim=16,
            decoder_l=2,
            threshold=0.1,
            decoder_arch="gcn",
            graph_prior=None,
            prior_mask=None,
            device=device,
            beta=1,
            learning_rate=0.0001,
            seed=np.random.randint(100),
        )
        print(gen.model)
        trainable_params = sum(p.numel() for p in gen.model.parameters())
        print("trainable_params", trainable_params)
        if not os.path.exists(f'tmp/{dname}{i}.pt'):
            gen.fit(X_train)
            shutil.copy(f'tmp/{dname}.pt', f'tmp/{dname}{i}.pt')
        gen.model.load_state_dict(torch.load(f'tmp/{dname}{i}.pt'), strict=False)

        start = time.time()
        sample(i, X_train, X_test, gen, 'val', dname)
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Required Time {:0>2}h {:0>2}m {:05.2f}s".format(int(hours), int(minutes), seconds))
        # sample(i, X_train, X_test, gen, 'train', dname)
        exit()



def sample(run, X_train, X_test, gen, split, dname):
    batchsize = 5
    if split == 'val':
        # X_test = X_test.iloc[0:batchsize+1]
        y = np.array(X_test['target'])
        X_num_shape = len([c for c in X_test.columns.tolist() if 'num' in c])
        X_cat = [np.array(X_test[c]) for c in X_test.columns.tolist() if 'cat' in c and 'cat0' not in c]
    else:
        # X_train = X_train.iloc[0:batchsize+1]
        y = np.array(X_train['target'])
        X_num_shape = len([c for c in X_train.columns.tolist() if 'num' in c])
        X_cat = [np.array(X_train[c]) for c in X_train.columns.tolist() if 'cat' in c and 'cat0' not in c]
    if len(X_cat)>0:
        X_cat = np.stack(X_cat, 1) 

    sample_gen = torch.zeros(y.shape[0], X_num_shape).cpu()
    remain_sample_id = torch.arange(len(sample_gen))
    if len(X_cat)>0:
        org_X_cat = torch.from_numpy(X_cat.astype(int))
    org_y = torch.from_numpy(y)
    max_iter = 10
    sample_iter = 0
    while len(remain_sample_id) > 0:
        if sample_iter > max_iter: break
        sample_iter += 1
        print(f'Remain {len(remain_sample_id)} Samples being generating')
        if len(X_cat)>0:
            X_cat = org_X_cat[remain_sample_id]
        y = org_y[remain_sample_id]
        # Create synthetic data
        if split == 'val':
            gen_data = gen.sample(X_test.iloc[0:batchsize+1])
        else:
            gen_data = gen.sample(X_train.iloc[0:batchsize+1])

        gen_y = np.array(gen_data['target'])
        # gen_X_cat = np.stack([np.array(gen_data[c]) for c in gen_data.columns.tolist() if 'cat' in c and 'cat0' not in c], 1) 

        X_num = np.stack([np.array(gen_data[c]) for c in gen_data.columns.tolist() if 'num' in c], 1) 

        X_num = torch.from_numpy(X_num.astype(float))
        # gen_X_cat = torch.from_numpy(gen_X_cat.astype(int))
        gen_y = torch.from_numpy(gen_y.astype(int))
        # batch_mask_cat_cond = torch.any(X_cat[:, None] != gen_X_cat[None, :], dim=-1).cpu() # N_y x N_sample
        batch_mask_y_cond = (y[:, None] != gen_y[None, :]).cpu() # N_y x N_sample
        # batch_mask_cond = torch.logical_or(batch_mask_y_cond, batch_mask_cat_cond)
        batch_mask_cond = batch_mask_y_cond
        mask_cond = batch_mask_cond.all(dim=1) # N_y
        all_yi = torch.where(~mask_cond)[0]
        yi_ls, samplei_ls = torch.where(~batch_mask_cond)
        sample_ind = []
        not_sample_ind = []
        for yi, samplei in zip(yi_ls, samplei_ls):
            if yi in all_yi:
                sample_ind.append(samplei)
                all_yi = all_yi[all_yi!=yi]
        
        sample_ind = torch.LongTensor(sample_ind)
        sample_gen[remain_sample_id[~mask_cond]] = X_num[sample_ind].float()
        remain_sample_id = remain_sample_id[mask_cond]
    if len(remain_sample_id) > 0:
        sample_gen[remain_sample_id[mask_cond]] = X_num[:len(remain_sample_id)].float()
    os.makedirs(f'tmp/{dname}{run}', exist_ok=True)
    np.save(f'tmp/{dname}{run}/X_num_{split}', sample_gen.numpy())



def load_data(dname):
    split = 'train'
    X_train = pd.read_csv(f'data/{dname}/{split}.csv')
    split = 'test'
    X_test = pd.read_csv(f'data/{dname}/{split}.csv')
    return X_train, X_test

if __name__  == '__main__':
    main()