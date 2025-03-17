import torch, os
from scipy.stats import wasserstein_distance_nd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import lib


class Evaluator:
    def metrics(self, all_samples, test_x):
        '''
        all_samples: sampled data
        test_x: test data (i.e., brain measurments)

        The input shapes of all_samples and test_x are 2D and both are same: batch x (num_visits x num_node)   
        '''
        # print(all_samples.shape, test_x.shape)
        # exit()
        if (all_samples.min() < -1e+10) or (all_samples.max() > 1e+10):
            print('sampled values are exploded !!!')
            wd, jsd_mean, rmse = None, None, None
        else:

            # (1) Wasserstein Distance 
            sampled_data_np = all_samples.cpu().numpy()
            real_data_np = test_x.cpu().numpy()

            wd = wasserstein_distance_nd(sampled_data_np, real_data_np)
            # print(f"\nWD: {wd:.3f}")

            # (2) Jensen-Shannon divergence
            sampled_data = F.softmax(all_samples, dim=1)
            real_data = F.softmax(test_x, dim=1)

            m = 0.5 * (sampled_data + real_data) 

            assert not torch.isnan(self.kld(sampled_data, m)).any().item(), "self.kld(sampled_data, m) has nan"

            jsd = 0.5 * (self.kld(sampled_data, m) + self.kld(real_data, m))
            jsd_mean = jsd.mean().item()
            
            # print(f"JSD: {jsd_mean:.3f}")

            # (3) RMSE
            mse = nn.MSELoss()
            mse = mse(all_samples, test_x)
            rmse = torch.sqrt(mse).item()
            # print(f"RMSE: {rmse:.3f}")

        return wd, jsd_mean, rmse


    def kld(self, p, q):
        p = p + 1e-10 
        return (p * (p.log() - q.log())).sum(dim=1)

from einops import rearrange


def evaluate(real_data_path, gen_data_path, SPLIT = 'train', dn='Amyloid'):
    x_gen = np.load(os.path.join(gen_data_path, f'X_num_{SPLIT}.npy'), allow_pickle=True)#[:, :-1]
    x_real = np.load(os.path.join(real_data_path, f'X_num_{SPLIT}.npy'), allow_pickle=True)#[:, :-1]
    status = SPLIT if SPLIT == 'train' else 'test'
    dir = f'../tab-ddpm/preprocessed/{dn}'
    raw_r = os.path.join(dir, status+'_data')
    visit_n = [torch.load(f'{raw_r}/{fn}').shape[0] for fn in os.listdir(raw_r)]
    x_gen, x_real = torch.from_numpy(x_gen), torch.from_numpy(x_real)
    assert sum(visit_n) == len(x_gen)
    x_gen = torch.tensor_split(x_gen, torch.LongTensor(visit_n).cumsum(0)[:-1])
    x_gen = torch.nn.utils.rnn.pad_sequence(x_gen, batch_first=True)
    x_gen = rearrange(x_gen, 'b v n -> b (v n)')
    x_real = torch.tensor_split(x_real, torch.LongTensor(visit_n).cumsum(0)[:-1])
    x_real = torch.nn.utils.rnn.pad_sequence(x_real, batch_first=True)
    x_real = rearrange(x_real, 'b v n -> b (v n)')
    eval = Evaluator()
    # print(SPLIT)
    # print("x_gen.shape, x_real.shape")
    # print(x_gen.shape, x_real.shape)
    wd, jsd_mean, rmse = eval.metrics(x_gen, x_real)
    # print(wd, jsd_mean, rmse)
    return wd, jsd_mean, rmse

methodn = 'goggle'
print(methodn)
for split in ['val', 'train']:
    for dn in ['Amyloid', 'CT', 'FDG', 'Tau']:
        wds, jsd_means, rmses = [], [], []
        for i in range(1, 4):
            real_data_path = f'../tab-ddpm/data/{dn}/'
            parent_dir = f'tmp/{dn}{i}'
            wd, jsd_mean, rmse = evaluate(real_data_path, parent_dir, split, dn)
            wds.append(wd)
            jsd_means.append(jsd_mean)
            rmses.append(rmse)

        print(dn, split)
        print(f'WD: {np.mean(wds):.5f} +- {np.std(wds):.5f}')
        print(f'JSD: {np.mean(jsd_means):.5f} +- {np.std(jsd_means):.5f}')
        print(f'RMSE: {np.mean(rmses):.5f} +- {np.std(rmses):.5f}')
