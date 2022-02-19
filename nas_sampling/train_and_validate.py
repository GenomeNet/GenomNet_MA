#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 21:10:39 2021

@author: amadeu
"""



import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint 
from  generalNAS_tools import utils

import darts_tools.architect

import gc

import logging
import sys
import torch.nn as nn
#log_format = '%(asctime)s %(message)s'


logging = logging.getLogger(__name__)




# def train(train_queue, valid_queue, model, rhn, conv, criterion, optimizer, optimizer_a, architect, unrolled, lr, epoch, num_steps, clip_params, report_freq, beta, one_clip=True, train_arch=True, pdarts=True):

def train(train_queue, random_model, criterion, optimizer, lr, epoch, rhn, conv, num_steps, clip_params, report_freq, beta, one_clip=True):
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()    # iterative_indices = np.random.permutation(self.iterative_indices)
    # train_queue = train_object
    total_loss = 0
    start_time = time.time()

    for step, (input, target) in enumerate(train_queue):
        
        if step > num_steps:
            break
        
        input, target = input, target
  
        random_model.train()
    
        input = input.float().to(device)#.cuda()
        
        target = torch.max(target, 1)[1]
        
        batch_size = input.size(0)

        
        
        target = target.to(device)#.cuda(non_blocking=True)
        
        hidden = random_model.init_hidden(batch_size) # [1,2,128]
        
      
        optimizer.zero_grad()
        
        logits, hidden, rnn_hs, dropped_rnn_hs = random_model(input, hidden, return_h=True)
        
        
        raw_loss = criterion(logits, target)

        loss = raw_loss
       
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        total_loss += raw_loss.data
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
            #logging.info(parallel_model.genotype())
            #logging.info(F.softmax(parallel_model.weights, dim=-1))
            logging.info('| step {:3d} | train obj {:5.2f} | '
                'train acc {:8.2f}'.format(step,
                                           objs.avg, top1.avg))
    return top1.avg, objs.avg
        
    
    
    
def evaluate_architecture(valid_queue, random_model, criterion, optimizer, epoch, report_freq, num_steps):
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
  
    random_model.eval()
    total_loss = 0
    # valid_queue = valid_object
    
    with torch.no_grad():
        
        for step, (input, target) in enumerate(valid_queue):
            
            if step > num_steps:
                break
            
            input, target = input.float(), target
            input = input.to(device)
            
            batch_size = input.size(0)


            target = torch.max(target, 1)[1]
            target = target.to(device)
            
            hidden = random_model.init_hidden(batch_size)
            
            logits, hidden = random_model(input, hidden)
            
            loss = criterion(logits, target)
            
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
            
            objs.update(loss.data, batch_size)
            top1.update(prec1.data, batch_size)
            
            if step % report_freq == 0:
                logging.info('| step {:3d} | valid obj {:5.2f} | '
                'valid acc {:8.2f}'.format(step,
                                           objs.avg, top1.avg))
            
    return top1.avg, objs.avg
