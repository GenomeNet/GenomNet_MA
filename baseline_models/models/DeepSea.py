#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:56:29 2021

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
import math


    
class NN_class(nn.Module):
    def __init__(self, num_classes, batch_size, seq_size, task):
        
        super(NN_class,self).__init__()
        
        self.layer1 = nn.Sequential(
                nn.Conv1d(4, 320, kernel_size=8),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=4, stride=4),
                nn.Dropout(p=0.2))
        
        self.layer2 = nn.Sequential(
                nn.Conv1d(320, 480, kernel_size=8),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=4, stride=4),
                nn.Dropout(p=0.2))
        
        self.layer3 = nn.Sequential(
                nn.Conv1d(480, 960, kernel_size=8),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5))
        
        neur1 = math.floor((((seq_size-8)+1)-4)/4+1) # 248
        neur2 = math.floor((((neur1-8)+1)-4)/4+1) # 59
        neur3 = math.floor(((neur2-8)+1))

        self.num_neurons = math.floor(neur3)
        self.fc1 = nn.Linear(960*self.num_neurons, 925)   #  141/2 ; 4
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(925, num_classes)
        
        if task == "TF_bindings":
            
            self.final = nn.Sigmoid()
            
        else:
            
            self.final = nn.Identity()
        
      
        
    def forward(self, x):
       

        x = self.layer1(x)  # results in 91 neurons and 6 channels
      
        x = self.layer2(x)
       
        x = self.layer3(x)
                
        x = torch.flatten(x, start_dim= 1) 
        
        
        x = self.fc1(x)
        
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.final(x)

        return x
