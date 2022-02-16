#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:06:32 2020

@author: amadeu
"""

import csv

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
from argparse import Namespace
import pandas as pd

import os

from numpy import array
import torch
import gc
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset,DataLoader

import math

# * 2 rr_blocks
# * every rr_block: conv1d, bn, relu, conv1d, bn
# * rest as in danQ (i.e. num_motifs, last fc layer etc., loss-function)
# * for residual_block always use padding, so number of units stays the same (original model needs padding of 1 for conv3x3, so number of units stays the same); whenever channel_size changes, use shortcut
# * stride can do downsampling

# * Since we only have 2 residual blocks, we bring conv channel size to 320 first and follow with residual blocks. This is as in ResNet where a conv layer comes before residual blocks as well.)
# * No down sampling in Residualblocks, since as described, the rest should be as in DanQ (where downsampling happens through max-pooling with stride=13)
# * the Projection-block would usually always use shortcut with 1x1 conv to increase channel size,  but at the same time it would do downsamplen with stride=2, since first conv layer of normalen layer has stride=2 as well;
#   in contrast, we do not have stride=2, since as in DanQ we only want to downsample through maxpooling


### Define ResidualBlock ###
def conv_26(in_channels, out_channels, kernel_size=26, stride=1, padding=12): # 
    return nn.Conv1d(in_channels, out_channels, kernel_size, 
                    stride, padding)

def conv_1(in_channels, out_channels, kernel_size=26, stride=1, padding=12): # 
    return nn.Conv1d(in_channels, out_channels, kernel_size, 
                    stride, padding)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Residual block
class ResidualBlock(nn.Module): # 
    def __init__(self, in_channels, expanded_channels, identity = False, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv_26(in_channels, expanded_channels, 26, 1, 12)
        self.bn1 = nn.BatchNorm1d(expanded_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv_26(expanded_channels, expanded_channels, 26, 1, 13)
        self.bn2 = nn.BatchNorm1d(expanded_channels)
        self.identity = identity
        if (self.identity == False): 
            self.shortcut = nn.Sequential(
                    conv_1(in_channels, expanded_channels, 1, 1, 0), 
                    nn.BatchNorm1d(expanded_channels))

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if (self.identity == False):
            residual = self.shortcut(residual)
        # print(x.shape)
        # print(residual.shape)
        x += residual
        x = self.relu(x)
        return x 
    


class NCNet_RR(nn.Module):
    def __init__(self, res_block, seq_size, num_classes, batch_size, task):
        super(NCNet_RR, self).__init__()
        self.num_classes = num_classes
        self.num_motifs = 320
        self.batch_size = batch_size
        self.conv1d = nn.Conv1d(4, self.num_motifs, kernel_size=26, stride=1) # with seq_len= 1000 kernel_size=26; seq_len=150 kernelsize=9
        self.seq_size = seq_size
        self.ResidualBlock = res_block
        
        self.bn = nn.BatchNorm1d(self.num_motifs)
        self.relu = nn.ReLU()
        self.block1 = self.ResidualBlock(self.num_motifs, self.num_motifs, identity = True)
        
        self.block2 = self.ResidualBlock(self.num_motifs, self.num_motifs, identity = True)

      
        self.maxpool = nn.MaxPool1d(kernel_size = 13, stride = 13) # seq_len=1000 kernel_size= 13 stride=13; seq_len=150 kernelsize=4 stride=4
        self.dropout = nn.Dropout(p=0.2)
            
        self.lstm = nn.LSTM(self.num_motifs, #
                            self.num_motifs,
                            bidirectional=True)
       
        
    
        aux_num = (((seq_size-26)+1)-13)/13+1
        self.num_neurons = math.floor(aux_num)

        self.dropout_2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.num_motifs*2*self.num_neurons, 925) # seq_len 1000 75neuronen; seq_len 150 35neuronen
        self.fc2 = nn.Linear(925, num_classes)
        
        if task == "TF_bindings":
            
            self.final = nn.Sigmoid()
            
        else:
            
            self.final = nn.Identity()
       

    def connect_blocks(self, res_block, in_channels, expanded_channels, num_blocks, stride=1):
       
        blocks = []
        blocks.append(res_block(in_channels, expanded_channels, identity = False)) 
        
        for i in range(1, num_blocks):
            blocks.append(res_block(expanded_channels, expanded_channels, identity = True))
        return nn.Sequential(*blocks) #

    def forward(self, x, batch_size):
        
        x = self.conv1d(x)
        #print('out') #with padding = 1 output remains 10 units

        x = self.bn(x)
        
        x = self.relu(x)
        #print('bn') # bn dimension stays constant

        x = self.block1(x)
        x = self.relu(x)

        x = self.block2(x)

      
        x = self.relu(x)
       
        x = self.maxpool(x)
        
        x = self.dropout(x)
        
        h_0, c_0 = self.zero_state(batch_size)
        # x.shape: [2,320,75]

        
        x = x.permute(2,0,1)

        #CNN expects [batchsize, input_channels, signal_length]
        # lstm expects shape [batchsize, signal_length, number of features]

        output, state = self.lstm(x, (h_0, c_0))
        
        # x.shape: [75,2,320]
        x = output.permute(1,0,2)
        # x.shape: [2,75,640]
        
        x = torch.flatten(x, start_dim= 1) 
        
        x = self.dropout_2(x)

        x = self.fc1(x)
      
        x = self.relu(x)
        
        x = self.fc2(x)
        
        x = self.final(x)
        
        return x
    
    def zero_state(self, batch_size):
        return (torch.zeros(2, batch_size, self.num_motifs).to(device),
               torch.zeros(2, batch_size, self.num_motifs).to(device))

