#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 07:51:45 2021

@author: amadeu
"""

import pickle
import mat73
import scipy.io
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np

def data_preprocessing(train_directory, valid_directory, test_directory, batch_size):
    
    data_dict_train = mat73.loadmat(train_directory)  # typically train.mat from DeepSea data
    inputs_train = data_dict_train["trainxdata"]
    targets_train = data_dict_train["traindata"]

    data_dict_val = scipy.io.loadmat(valid_directory)  # typically valid.mat from DeepSea data
    inputs_val = data_dict_val["validxdata"]
    targets_val = data_dict_val["validdata"]

    data_dict_test = scipy.io.loadmat(test_directory)  # typically test.mat from DeepSea data
    inputs_test = data_dict_test["testxdata"]
    targets_test = data_dict_test["testdata"]
    
    inputs_train, targets_train = torch.Tensor(inputs_train), torch.Tensor(targets_train)
    inputs_train, targets_train = inputs_train.float(), targets_train.float()
    
    inputs_val, targets_val = torch.Tensor(inputs_val), torch.Tensor(targets_val)
    inputs_val, targets_val = inputs_val.float(), targets_val.float()
    
    inputs_test, targets_test = torch.Tensor(inputs_test), torch.Tensor(targets_test)
    inputs_test, targets_test = inputs_test.float(), targets_test.float()

    
    
    class get_data(Dataset):
        def __init__(self,feature,target):
            self.feature = feature
            self.target = target
        def __len__(self):
            return len(self.feature)
        def __getitem__(self,idx):
            item = self.feature[idx]
            label = self.target[idx]
            return item,label
        
    
    train_feat = []
    for batch in inputs_train:
        train_feat.append(batch)
        
    train_targ = []
    for batch in targets_train:
        train_targ.append(batch)
      
        
    val_feat = []
    for batch in inputs_val:
        val_feat.append(batch)
        
    val_targ = []
    for batch in targets_val:
        val_targ.append(batch)
        
        
    test_feat = []
    for batch in inputs_test:
        test_feat.append(batch)
        
    test_targ = []
    for batch in targets_test:
        test_targ.append(batch)
        
    train = get_data(train_feat, train_targ)# 
    valid = get_data(val_feat, val_targ)
    test = get_data(test_feat, test_targ)

    
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)  # shuffle ensures random choices of the sequences are used
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle=False)

    
    return train_loader, valid_loader, test_loader
