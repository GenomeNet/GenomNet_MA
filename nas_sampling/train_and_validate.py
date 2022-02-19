#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 21:10:39 2021

@author: amadeu
"""

import gc

import logging
import torch
import torch.nn as nn

from  generalNAS_tools import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


logging = logging.getLogger(__name__)

def train(train_queue, random_model, criterion, optimizer, lr, epoch, rhn, conv, num_steps, clip_params, report_freq, beta, one_clip=True):
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        
        if step > num_steps:
            break
        
        input, target = input, target
  
        random_model.train()
    
        input = input.float().to(device)#.cuda()
        
        target = torch.max(target, 1)[1]
        
        batch_size = input.size(0)

        
        
        target = target.to(device)
        
        hidden = random_model.init_hidden(batch_size) # [1,2,128]
        
      
        optimizer.zero_grad()
        
        logits, hidden, rnn_hs, dropped_rnn_hs = random_model(input, hidden, return_h=True)
        
        
        raw_loss = criterion(logits, target)

        loss = raw_loss
       
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()
      
        gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            
        if one_clip == True:
            torch.nn.utils.clip_grad_norm_(random_model.parameters(), clip_params[2])
        else:
            torch.nn.utils.clip_grad_norm_(conv, clip_params[0])
            torch.nn.utils.clip_grad_norm_(rhn, clip_params[1])
            
        optimizer.step()
        
        prec1,prec2 = utils.accuracy(logits, target, topk=(1,2)) 
                
        objs.update(loss.data, batch_size)
       
        top1.update(prec1.data, batch_size) 

        if step % report_freq == 0 and step > 0:
            logging.info('| step {:3d} | train obj {:5.2f} | '
                'train acc {:8.2f}'.format(step,
                                           objs.avg, top1.avg))
    return top1.avg, objs.avg
        
