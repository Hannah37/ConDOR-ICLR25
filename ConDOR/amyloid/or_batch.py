import os
import pandas as pd
import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt

from einops import rearrange, reduce, repeat
from statsmodels.miscmodels.ordinal_model import OrderedModel

class OrdinalRegression():
    def __init__(self, args):
        self.df = pd.read_csv(os.path.join(args.dir, 'Amyloid.csv'))
        self.age = self.df['age']
        self.label = self.df['label']
        self.data = self.df.iloc[:, -148:]
        self.data = round(self.data, 4) 
        self.x_columns = ['Node ' + str(i) for i in range(1, 149)]
        self.a_tol = args.age_tolerance
        self.num_node = args.num_node
        self.args = args
        self.df['age_norm'] = (self.df['age']- args.age_min) / (args.age_max - args.age_min)
        self.age = self.df['age_norm']

        # fitting ordinal regression model
        self.model = OrderedModel(self.label, self.df[['age_norm'] + self.x_columns], distr='logit')
        result = self.model.fit(method='bfgs')
        self.params = result.params
        self.epsilon = 1e-5

        self.n_thresholds = args.classes - 1
        print('n_thresholds: ', self.n_thresholds)

        self.threshold = self.model.transform_threshold_params(result.params[-self.n_thresholds:])
        print('Threshold: ', self.threshold)

    def logistic_cdf(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def get_device(self, device):
        self.device = device
        self.tensor_data = torch.tensor(self.data.to_numpy(), dtype=torch.float32).to(self.device)
        self.tensor_a = torch.tensor(self.age.to_numpy(), dtype=torch.float32).to(self.device)
        self.tensor_y = torch.tensor(self.label.to_numpy(), dtype=torch.float32).to(self.device)
        self.threshold = torch.tensor(self.threshold).to(self.device)

        self.initialize_x_values(self.args.batch)

    # P(Y=y|X, A=a) is fitted with ordinal regression
    def p_y_given_x_a(self, x, y, a):
        batch_size = a.size(0)
        num_features = x.size(1) 

        if a.dim() == 1:
            a = a.unsqueeze(1)  

        x = x.unsqueeze(1) 
        a = a.unsqueeze(2)  

        _exog = torch.cat([a, x], dim=2).float().to(self.device) 

        _params = torch.tensor(self.params, device=self.device).float().unsqueeze(0) 
        expected_params = num_features + 1 
        _params = _params[..., :-self.n_thresholds].unsqueeze(1)  

        xb = torch.matmul(_exog, _params.transpose(1, 2)).squeeze()  
        low = self.threshold[y] - xb
        upp = self.threshold[y + 1] - xb

        py = self.logistic_cdf(upp) - self.logistic_cdf(low) 
        
        assert (py >= 0).all(), "Error: P(Y|X, A) < 0"
        return py

    # P(X|A=a) is estimated using KDE
    def kde_density(self, data, x_values, mask, bandwidth=0.001):
        data = data.reshape(data.size(0), -1, 1)  
        x_values = x_values.reshape(x_values.size(0), 1, -1)  
        diffs = x_values - data  # [batch_size, n_data_points, 100]

        # Apply Gaussian kernel
        bandwidth_tensor = torch.tensor(bandwidth, device=data.device)
        kernel_vals = torch.exp(-0.5 * (diffs / bandwidth_tensor) ** 2) / (bandwidth_tensor * torch.sqrt(torch.tensor(2 * torch.pi, device=data.device)))

        # Apply mask to kernel values
        mask = mask.reshape(mask.size(0), -1, 1)  
        masked_kernel_vals = kernel_vals * mask  # Apply mask

        valid_counts = mask.sum(dim=1)  
        density = masked_kernel_vals.sum(dim=1) / valid_counts 
        density /= density.sum(dim=1, keepdim=True) 
        return density

    def initialize_x_values(self, batch_size):
        min_vals = self.tensor_data.min(dim=0).values
        max_vals = self.tensor_data.max(dim=0).values
        steps = self.args.x_lin_steps
        x_values = torch.zeros((batch_size, self.num_node, steps), device=self.device)

        for node_index in range(self.num_node):
            linspace = torch.linspace(min_vals[node_index], max_vals[node_index], steps, device=self.device)
            x_values[:, node_index, :] = linspace.repeat(batch_size, 1)

        self.x_values = x_values
    
    def get_cdf(self, y, a, a_tol):
        a_expanded = a.unsqueeze(1)  
        is_close_mask = torch.isclose(self.tensor_a.unsqueeze(0), a_expanded, atol=a_tol)

        all_data = [self.tensor_data[is_close_mask[i], :] for i in range(a.size(0))]  
        max_length = max([d.size(0) for d in all_data])  
        padded_data = torch.zeros((len(all_data), self.num_node, max_length), device=self.device)
        curr_batch = padded_data.shape[0]
        mask = torch.zeros_like(padded_data, dtype=torch.bool)

        for i, data in enumerate(all_data):
            for node_index in range(self.num_node):
                actual_length = data.size(0)
                padded_data[i, node_index, :actual_length] = data[:, node_index]
                mask[i, node_index, :actual_length] = 1

        p_x_given_a = []
        for node_index in range(self.num_node):
            node_x_values = self.x_values[:, node_index, :]  
            node_x_values = node_x_values[:curr_batch, :]
            node_data = padded_data[:, node_index, :]  
            node_mask = mask[:, node_index, :]  
            density = self.kde_density(node_data, node_x_values, node_mask)
            p_x_given_a.append(density)

        p_x_given_a = torch.stack(p_x_given_a, dim=1)  

        # P(Y=y|A=a)
        age_mask = (self.tensor_a >= (a.unsqueeze(1) - a_tol)) & (self.tensor_a <= (a.unsqueeze(1) + a_tol))
        y = torch.argmax(y, dim=1)
        label_mask = (self.tensor_y.unsqueeze(0) == y.unsqueeze(1))
        p_y_given_a = torch.sum(age_mask * label_mask, dim=1).float() / (torch.sum(age_mask, dim=1).float() + self.epsilon)

        p_y_given_x_a_list = []
        for i in range(self.args.x_lin_steps):
            step_x_values = self.x_values[:curr_batch, :, i]  
            p_y_given_x_a = self.p_y_given_x_a(step_x_values, y, a_expanded) 
            p_y_given_x_a_list.append(p_y_given_x_a)

        p_y_given_x_a = torch.stack(p_y_given_x_a_list, dim=1) 

        p_x_given_a_y = (p_y_given_x_a.unsqueeze(1) * p_x_given_a) / (p_y_given_a.unsqueeze(1).unsqueeze(2) + self.epsilon)
        
        for i in range(self.num_node):
            x_step = self.x_values[0, i, 1] - self.x_values[0, i, 0]
            p_x_given_a_y[:, i, :] /= torch.sum(p_x_given_a_y[:, i, :] * x_step, dim=1, keepdim=True)

        cdf_list = [
        torch.cumsum(
            p_x_given_a_y[:, i, :] * (self.x_values[0, i, 1] - self.x_values[0, i, 0]), 
            dim=1
        ) 
        for i in range(self.num_node)]

        cdf = torch.stack(cdf_list, dim=1) 
        return cdf
    
    def sample_from_cdf(self, y, a, num_samples=1):
        cdf = self.get_cdf(y, a, self.a_tol) 
        cdf_steps = self.args.x_lin_steps

        uniform_samples = torch.rand(num_samples, self.num_node).to(self.device)
        uniform_samples = uniform_samples.unsqueeze(0).expand(cdf.shape[0], -1, -1) 
        uniform_samples = uniform_samples.transpose(1, 2) 

        cdf_indices = torch.searchsorted(cdf, uniform_samples, right=True) 
        cdf_indices = torch.clamp(cdf_indices, 0, cdf_steps - 1)

        batch_indices = torch.arange(cdf.shape[0]).unsqueeze(1).unsqueeze(2).expand(-1, cdf.shape[1], num_samples)
        node_indices = torch.arange(cdf.shape[1]).unsqueeze(0).unsqueeze(2).expand(cdf.shape[0], -1, num_samples)

        sampled_values = self.x_values[batch_indices, node_indices, cdf_indices]  
        sampled_values = sampled_values.transpose(1, 2) 
        
        assert not (sampled_values < 0).any(), "The sampled X contains negative values."
        assert not torch.isnan(sampled_values).any(), "The sampled X contains nan values."
        assert not torch.isinf(sampled_values).any(), "The sampled X contains inf values."

        return sampled_values