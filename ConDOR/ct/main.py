import os
import sys
import time
import wandb
import warnings
import argparse
import numpy as np
import torch
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'MODEL'))
from MODEL import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D, warmup_Dataset1D

warnings.filterwarnings("ignore")


def run(args, current_time):
    if args.warmup == True:
        train_dataset = warmup_Dataset1D(args, 'train') 
        test_dataset = warmup_Dataset1D(args, 'test') 
        args.test_num = len(test_dataset)
    else:
        test_dataset = Dataset1D(args, 'test') 
        train_dataset = Dataset1D(args, 'train') 
        
    print("# train data: ", len(train_dataset))
    print("# test data: ", len(test_dataset))
    max_visit = train_dataset.max_visit()

    '''model initialization'''
    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        n_classes = args.classes,
        max_visit = max_visit
    )

    diffusion = GaussianDiffusion1D(
        model,
        num_node = args.num_node,
        timesteps = args.timesteps,
        objective = 'pred_noise',
        norm_min = args.data_min,
        norm_max = args.data_max,
        args = args,
    )


    trainer = Trainer1D(
        diffusion,
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        train_batch_size = args.batch,
        test_size = args.test_num,
        train_lr = args.lr,
        train_num_steps = args.train_num_steps,  # total training steps for all visitis
        warmup_num_steps = args.warmup_num_steps,   # warm-up training steps for the first visit
        gradient_accumulate_every = args.gradient_accumulate_every,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        save_and_sample_every = args.sampling_epoch, # sampling cycle
        results_folder = './results/' + current_time, # folder to save samples
        amp = True,                       # turn on mixed precision
        optim = args.optim,
        norm_min = args.data_min,
        norm_max = args.data_max,
        warmup = args.warmup,
        alpha = args.alpha,
    )

    '''training'''
    trainer.train()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters) 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup', type=int, default=0, help='set 1 if you need warmup')
    parser.add_argument('--batch', type=int, default=16) 
    parser.add_argument('--train_num_steps', type=int, default=10000) 
    parser.add_argument('--warmup_num_steps', type=int, default=10000) 
    parser.add_argument('--timesteps', type=int, default=1000) 
    parser.add_argument('--sampling_epoch', type=int, default=50) 
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, default='Adam') 
    parser.add_argument('--num_node', type=str, default=148) 
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--start_time', type=float, default=time.time(), help='start time of training')

    '''Cortical Thickness'''
    parser.add_argument('--dataset', type=str, default='CT') 
    parser.add_argument('--alpha', type=float, default=0.1, help='weight of cohort-level noise, range: (0, 1)')
    parser.add_argument('--dir', type=str, default='/home/user9/Condor/data/CT') 
    parser.add_argument('--data_min', type=float, default=1.027, help='minimum value of CT')
    parser.add_argument('--data_max', type=float, default=4.81, help='maximum value of CT')
    parser.add_argument('--age_min', type=float, default=55.0, help='minimum value of CT age')
    parser.add_argument('--age_max', type=float, default=92.3, help='maximum value of CT age')
    parser.add_argument('--test_num', type=int, default=36, help='number of test data') 
    parser.add_argument('--x_lin_steps', type=int, default=100, help='X linespace steps') 
    parser.add_argument('--gradient_accumulate_every', type=int, default=1, help='loss update for n data')
    parser.add_argument('--age_tolerance', type=float, default=0.03) # 0.03=1+@
    
    args = parser.parse_args()
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed)

    current_time = time.strftime('%B%d_%H_%M_%S', time.localtime(time.time()))
    print('current_time: ', current_time)

    wandb.init(project="Condor", allow_val_change=True)    
    wandb.run.name = current_time 
    wandb.run.save() 
    wandb.config.update(args)

    start = args.start_time
    run(args, current_time)
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training Time {:0>2}h {:0>2}m {:05.2f}s".format(int(hours), int(minutes), seconds))