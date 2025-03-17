from ctgan import CTGAN
import torch
import pandas as pd
import numpy as np
import sys, os
import time

device = 'cuda:0'
def main():
    X_train, X_test = load_data(sys.argv[1])

    # Names of the columns that are discrete
    discrete_columns = [c for c in X_train.columns.tolist() if 'cat' in c]
    # print(X_train)
    for i in range(3):
        ctgan = CTGAN(epochs=10)
        ctgan.fit(X_train, discrete_columns)
        print(ctgan._generator)
        trainable_params = sum(p.numel() for p in ctgan._generator.parameters())
        print("trainable_params", trainable_params)
        print('train')
        sample(i+1, X_train, X_test, ctgan, 'train')
        print('val')
        start = time.time()
        sample(i+1, X_train, X_test, ctgan, 'val')
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Required Time {:0>2}h {:0>2}m {:05.2f}s".format(int(hours), int(minutes), seconds))

def load_data(dname):
    split = 'train'
    X_train = pd.read_csv(f'../goggle/data/{dname}/{split}.csv')
    split = 'test'
    X_test = pd.read_csv(f'../goggle/data/{dname}/{split}.csv')
    return X_train, X_test

def sample(run, X_train, X_test, ctgan, split):
    batch_size = 1000
    if split == 'val':
        y = np.array(X_test['target'])
        X_num_shape = len([c for c in X_test.columns.tolist() if 'num' in c])
        X_cat = np.stack([np.array(X_test[c]) for c in X_test.columns.tolist() if 'cat' in c], 1) 
    else:
        y = np.array(X_train['target'])
        X_num_shape = len([c for c in X_train.columns.tolist() if 'num' in c])
        X_cat = np.stack([np.array(X_train[c]) for c in X_train.columns.tolist() if 'cat' in c], 1) 

    sample_gen = torch.zeros(y.shape[0], X_num_shape).cpu()
    remain_sample_id = torch.arange(len(sample_gen))
    org_X_cat = torch.from_numpy(X_cat.astype(int)).to(device)
    org_y = torch.from_numpy(y).to(device)
    max_iter = 10
    sample_iter = 0
    while len(remain_sample_id) > 0:
        if sample_iter > max_iter: break
        sample_iter += 1
        print(f'Remain {len(remain_sample_id)} Samples being generating')
        X_cat = org_X_cat[remain_sample_id.to(device)]
        y = org_y[remain_sample_id.to(device)]
        # Create synthetic data
        gen_data = ctgan.sample(batch_size)

        gen_y = np.array(gen_data['target'])
        gen_X_cat = np.stack([np.array(gen_data[c]) for c in gen_data.columns.tolist() if 'cat' in c], 1) 

        X_num = np.stack([np.array(gen_data[c]) for c in gen_data.columns.tolist() if 'num' in c], 1) 

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

    os.makedirs(f'CTGAN/generate_sample_ctgan{run}/{sys.argv[1]}', exist_ok=True)
    np.save(f'CTGAN/generate_sample_ctgan{run}/{sys.argv[1]}/X_num_{split}', sample_gen.numpy())


if __name__ == '__main__':
    main()