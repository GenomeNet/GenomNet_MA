#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 12:54:39 2020

@author: amadeu
"""


import csv

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
from argparse import Namespace
import pandas as pd

import numpy as np 
import pandas as pd 
import os

from numpy import array
import torch
import gc
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset,DataLoader
#from pytorch_model_summary import summary
 



   
class NN_class(nn.Module):
    def __init__(self, num_classes, num_motifs, batch_size):
        super(NN_class,self).__init__()
        
        self.conv1d = nn.Conv1d(4,num_motifs,kernel_size=10) # with sequence length 150; kernel=10; with sequence length 10 and kernel=3
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size = 2) # units=141; 8
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(num_motifs*70,num_motifs)   #  141/2 ; 4
        self.fc2 = nn.Linear(num_motifs,num_classes)
        
    def forward(self, x):
       
        #print(x.shape)
        x = self.conv1d(x)# results in 91 units and 6 channels
      
        #print(x.shape)
        x = self.relu(x)
       
        x = self.maxpool(x) #
        #print(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
      
        #print(x.shape)
        x = self.dropout(x)
        x = self.fc1(x)
        #print('dense_1')
        #print(x.shape)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#class NN_class(nn.Module):
#    def __init__(self, num_classes, num_motifs, batch_size):
#        super(NN_class, self).__init__()
#        self.conv1 = nn.Conv1d(4, num_motifs, 10)
#        self.pool = nn.MaxPool1d(5, stride = 2)
#        self.conv2 = nn.Conv1d(num_motifs, 200, 10)
#        self.fc1 = nn.Linear(200 * 16, 120)
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, 4)
#
#    def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = self.pool(x)
#        #print(x.shape)
#        x = F.relu(self.conv2(x))
#        x = self.pool(x)
#        #print('x2')
#        #print(x.shape)
#        x = x.view(-1, 200 * 16)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
 #       x = self.fc3(x)
 #       return x



##
#from modelsummary import summary
#model = DeepVirFinder()
#print(summary(model, torch.zeros((32,4,100)), show_input=False)) # as CNN expects shape of [batchsize, input_channel, signal_length]







































