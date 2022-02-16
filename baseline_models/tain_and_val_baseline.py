#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 12:20:31 2021

@author: amadeu
"""


import os
import sys
import time
import glob
import numpy as np
import torch
# import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import gc

from generalNAS_tools import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import csv
import gc
#import DeepVirFinder_model as dvf
import generalNAS_tools.data_preprocessing_new as dp
import torch
#from DeepVirFinder_model import model, device, criterion, optimizer, num_motifs, num_classes
#from data_preprocessing import train_loader, valid_loader, batch_size, num_batches_valid, num_batches_train, valid_y

import torch.nn as nn
import numpy as np 
from sklearn.metrics import classification_report
# precision, recall, f1
from sklearn.metrics import f1_score, recall_score, precision_score
# ROC PR, ROC AUC
from sklearn.metrics import roc_auc_score, auc

from sklearn.metrics import precision_recall_curve
from generalNAS_tools.utils import scores_perClass, scores_Overall, pr_aucPerClass, roc_aucPerClass, overall_acc, overall_f1

import matplotlib.pyplot as plt


# train_loader, num_steps = train_queue, 3
def Train(model, train_loader, optimizer, criterion, device, num_steps, task):
        
    objs = utils.AvgrageMeter()
 
    total_loss = 0
    start_time = time.time()
    
    labels = []
    predictions = []
    scores = nn.Softmax()
    
    for idx, (inputs, targets) in enumerate(train_loader):
        
        if idx > num_steps:
            break
        
        input, label = inputs, targets
       
        model.train()

        input = input.float().to(device)#.cuda()
        
        batch_size = input.size(0)
        
        optimizer.zero_grad()
        
        label = label.to(device)#.cuda(non_blocking=True)
        logits = model(input.float(), batch_size) #, (state_h, state_c))

        loss = criterion(logits, label)#.long()) 
     
        labels.append(label.detach().cpu().numpy())
        if task == "next_character_prediction":
            predictions.append(scores(logits).detach().cpu().numpy())
        else:#if args.task == "TF_bindings":
            predictions.append(logits.detach().cpu().numpy())

        loss.backward()
        
        optimizer.step()
                
        objs.update(loss.data, batch_size)
       
    return labels, predictions, objs.avg.detach().cpu().numpy()
    



def Valid(model, valid_loader, optimizer, criterion, device, num_steps, task):
   
    objs = utils.AvgrageMeter()
    #top1 = utils.AvgrageMeter()
    #top5 = utils.AvgrageMeter()
    
    total_loss = 0
    start_time = time.time()
    
    model.eval()
    
    labels = []
    predictions = []
    scores = nn.Softmax()


    with torch.no_grad():
        
        for idx, (inputs, targets) in enumerate(valid_loader):
            
            
            if idx > num_steps:
                break
        
            input, label = inputs, targets
               
            input = input.float().to(device)#.cuda()
            
            #label = torch.max(label, 1)[1]
            label = label.to(device)#.cuda(non_blocking=True)
            batch_size = input.size(0)
        
            
            #if args.task == "TF_bindings":
            logits = model(input.float(), batch_size) #, (state_h, state_c))

            loss = criterion(logits, label)
            #else:
                # label = torch.max(label, 1)[1]
            #logits = model(input.float()) #, (state_h, state_c))
            
            labels.append(label.detach().cpu().numpy())
            if task == "next_character_prediction":
                predictions.append(scores(logits).detach().cpu().numpy())
            else:
                predictions.append(logits.detach().cpu().numpy())
                
            objs.update(loss.data, batch_size)
  
    return labels, predictions, objs.avg.detach().cpu().numpy() #top1.avg, objs.avg




def Test(model, test_queue, optimizer, criterion, device, num_steps, task):
    
    objs = utils.AvgrageMeter()
    #top1 = utils.AvgrageMeter()
    #top5 = utils.AvgrageMeter()
    
    total_loss = 0
    start_time = time.time()
    
    model.eval()

    labels = []
    predictions = []
    scores = nn.Softmax()

    with torch.no_grad():
        
        for idx, (inputs, targets) in enumerate(test_queue):
        
            input, label = inputs, targets
               
            input = input.float().to(device)#.cuda()
            
            #label = torch.max(label, 1)[1]
            label = label.to(device)#.cuda(non_blocking=True)
            batch_size = input.size(0)
        
            #if args.task == "TF_bindings":
            logits = model(input.float(), batch_size) #, (state_h, state_c))

            loss = criterion(logits, label)
            #else:
               # label = torch.max(label, 1)[1]
            #logits = model(input.float()) #, (state_h, state_c))
            
            labels.append(label.detach().cpu().numpy())
            if task == "next_character_prediction":
                predictions.append(scores(logits).detach().cpu().numpy())
            else:
                predictions.append(logits.detach().cpu().numpy())
                
            objs.update(loss.data, batch_size)
            
            
    return labels, predictions, objs.avg.detach().cpu().numpy() #top1.avg, objs.avg

            
            
