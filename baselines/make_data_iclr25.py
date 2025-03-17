import os
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np

def load_data(status, dname='Amyloid'):
    CLASS_N = {
        'Amyloid': 5,
        'CT': 5,
        'FDG': 5,
        'Tau': 5
    }
    dir = f'preprocessed/{dname}'
    data_list, label_list = [], []
    cat_list = []
    filename_list = []
    for filename in os.listdir(os.path.join(dir, status+'_data')):
        data = torch.load(os.path.join(dir, status+'_data', filename))

        if data.shape[1] == 1:
            print(filename, 'The number of visit is 1')
            break
        filename_list.append(filename)
        data_list.append(data)

        label = torch.load(os.path.join(dir, status+'_label', filename))
        label_list.append(label) 


        age = torch.load(os.path.join(dir, status+'_age', filename))
        age_ = age.round().long()

        if (len(label) != len(age)) or (len(label) != data.shape[0]):
            print(filename, data.shape, len(label), len(age))
            print("Length of label, age, data should be same")

        assert (len(label) == len(age)) and (data.shape[0] == len(label)), "Length of label, age, data should be same"

        cat = age_[:, None]
        if dname == 'Tau':
            datatype = torch.load(os.path.join(dir, status+'_datatype', filename))
            datatype = torch.cat([datatype for _ in range(len(label))])
            cat = torch.stack([age_, datatype], 1)
        cat_list.append(cat)
        
    return data_list, label_list, cat_list

def make_data(status = 'test', dname = 'Amyloid'):

    data_list, label_list, cat_list = load_data(status, dname=dname)

    saver = f'data/{dname}' 
    os.makedirs(saver, exist_ok=True)
    data_list = torch.cat(data_list)
    label_list = torch.cat(label_list)
    cat_list = torch.cat(cat_list)
    X_cat = cat_list.numpy().astype(str)
    X_num = data_list.numpy()
    # X_num = torch.cat([data_list, age_list[:, None]], dim=1).numpy()
    y = label_list.numpy()

    print(status, X_num.shape, y.shape, X_cat.shape)
    print(X_num.min(), y.min())
    print(X_num.max(), y.max())
    np.save(f'{saver}/X_cat_{status}.npy', X_cat, allow_pickle=True)
    np.save(f'{saver}/X_num_{status}.npy', X_num, allow_pickle=True)
    np.save(f'{saver}/y_{status}.npy', y, allow_pickle=True)
    if status == 'test':
        status = 'val'
        np.save(f'{saver}/X_cat_{status}.npy', X_cat, allow_pickle=True)
        np.save(f'{saver}/X_num_{status}.npy', X_num, allow_pickle=True)
        np.save(f'{saver}/y_{status}.npy', y, allow_pickle=True)


dnames = os.listdir('preprocessed')
for dn in dnames:
    print(dn)
    make_data('train', dn)
    make_data('test', dn)
