#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 20:50:56 2021

@author: amadeu
"""


import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint 
from  generalNAS_tools import utils

import gc

import logging
import sys
import torch.nn as nn
from generalNAS_tools.custom_loss_functions import sample_weighted_BCE


logging = logging.getLogger(__name__)

def train(train_queue, valid_queue, model, rhn, conv, criterion, optimizer, optimizer_a, architect, unrolled, lr, epoch, num_steps, clip_params, report_freq, beta, one_clip=True, train_arch=True, pdarts=True, task=None):
    objs = utils.AvgrageMeter()
    
    total_loss = 0
    start_time = time.time()
    scores = nn.Softmax()

    labels = []
    predictions = []
         
    for step, (input, target) in enumerate(train_queue): 
        
        if step > num_steps:
            break        
        
        input, target = input, target
       
        model.train()

        input = input.float().to(device)
        target = target.to(device)
        batch_size = input.size(0)
        
        hidden = model.init_hidden(batch_size) 
        
        if train_arch: 
            
            try:
                input_search, target_search = next(valid_queue_iter)
                input_search = input_search.float()

            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
                input_search = input_search.float()
                
            input_search = input_search.to(device)
            target_search = target_search.to(device)
            
            if pdarts:
                optimizer_a.zero_grad()
                
                # rnn_hs is output from RHN/LSTM layer before dropout and linear layer; dropped_rnn_hs is output from 
                # RHN/LSTM after LSTM and dropout but befor linear layer
                # this uses this tensor for loss regularization (penalize big differences between different batches)
                # as we don't have time dependecy between batches, we don't need this part
                logits, hidden, rnn_hs, dropped_rnn_hs = model(input_search, hidden, return_h=True)
    
                loss_a = criterion(logits, target_search)
                loss_a.backward()
                nn.utils.clip_grad_norm_(model.arch_parameters(), clip_params[2]) 
                # Darts makes no clipping in the CNN, in the RHN they use clipping param of 0.25;
                # Pdarts uses clip param of 5 for CNN
                optimizer_a.step()
            else:
                hidden_valid = model.init_hidden(batch_size) 

                hidden_valid, grad_norm = architect.step( 
                                            hidden, input, target,
                                            hidden_valid, input_search, target_search,
                                            optimizer,
                                            unrolled)
                

        optimizer.zero_grad()
        
        hidden = model.init_hidden(batch_size)
        
        #with profiler.profile(with_stack=True, profile_memory=True) as prof:
        #with record_function("model_inference"):

        logits, hidden, rnn_hs, dropped_rnn_hs = model(input, hidden, return_h=True)
        
        raw_loss = criterion(logits, target)
        loss = raw_loss
       
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        total_loss += raw_loss.data
        loss.backward()
      
        gc.collect()

        if one_clip == True:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_params[2])
        else:
            torch.nn.utils.clip_grad_norm_(conv, clip_params[0])
            torch.nn.utils.clip_grad_norm_(rhn, clip_params[1])
            
        optimizer.step()
        
        objs.update(loss.data, batch_size)
        
        labels.append(target.detach().cpu().numpy())
        
        if task == "next_character_prediction":
            predictions.append(scores(logits).detach().cpu().numpy())
        else: #if args.task == "TF_bindings"::
            predictions.append(logits.detach().cpu().numpy())
       
        if step % report_freq == 0 and step > 0:
        
            logging.info('| step {:3d} | train obj {:5.2f}'.format(step, objs.avg))
            
    return labels, predictions, objs.avg.detach().cpu().numpy() # top1.avg, objs.avg


def infer(valid_queue, model, criterion, batch_size, num_steps, report_freq, task=None):
    objs = utils.AvgrageMeter()

    model.eval()
    
    total_loss = 0
    
    labels = []
    predictions = []
    
    scores = nn.Softmax()


    for step, (input, target) in enumerate(valid_queue):
        
        if step > num_steps:
            break
        
        input = input.to(device).float()
        batch_size = input.size(0)

        target = target.to(device)
        hidden = model.init_hidden(batch_size)#.to(device)  

        with torch.no_grad():
            logits, hidden = model(input, hidden)

            loss = criterion(logits, target)

        
        objs.update(loss.data, batch_size)
        labels.append(target.detach().cpu().numpy())
        if task == "next_character_prediction":
            predictions.append(scores(logits).detach().cpu().numpy())
        else:#if args.task == "TF_bindings"::
            predictions.append(logits.detach().cpu().numpy())

        if step % report_freq == 0:
            logging.info('| step {:3d} | val obj {:5.2f}'.format(step, objs.avg))


    return labels, predictions, objs.avg.detach().cpu().numpy()
