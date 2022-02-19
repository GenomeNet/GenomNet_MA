#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 10:35:19 2021

@author: amadeu
"""

import torch
import time

from generalNAS_tools.train_and_validate import train, infer

from generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint

from generalNAS_tools import utils

from generalNAS_tools.utils import scores_perClass, scores_Overall, pr_aucPerClass, roc_aucPerClass, overall_acc, overall_f1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import gc
#from torch.profiler import profile, record_function, ProfilerActivity

import logging
import sys
import torch.nn as nn

import numpy as np

logging = logging.getLogger(__name__)


def hb_step(train_queue, valid_queue, random_model, rhn, conv, criterion, scheduler, batch_size, optimizer, optimizer_a, architect, unrolled, num_steps, clip_params, report_freq, beta, one_clip, task, budget):
    
    train_losses = []
    valid_losses = []
    
    train_acc = []
    valid_acc = []
    
    for epoch in range(budget): 
        # epoch=0
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        # supernet.drop_path_prob = args.drop_path_prob * epoch / self.epochs
        # supernet.drop_path_prob = drop_path_prob * epoch / epochs
                                              
        epoch_start = time.time()
        
        lr = scheduler.get_last_lr()[0]
    
        labels, predictions, train_loss = train(train_queue, valid_queue, random_model, rhn, conv, criterion, optimizer, None, None, unrolled, lr, epoch, num_steps, clip_params, report_freq, beta, one_clip, train_arch=False, pdarts=False, task=task)
        
        scheduler.step()

        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        train_losses.append(train_loss)

    
        if task == 'next_character_prediction':
            acc = overall_acc(labels, predictions, task)
            logging.info('| epoch {:3d} | train acc {:5.2f}'.format(epoch, acc))
            train_acc.append(acc)
        else:
            f1 = overall_f1(labels, predictions, task)
            logging.info('| epoch {:3d} | train f1-score {:5.2f}'.format(epoch, f1))
            train_acc.append(f1)
    
    
        # valid_acc, valid_obj = evaluate_architecture(valid_object, random_model, criterion, optimizer, epoch, args.report_freq, args.num_steps) 
        labels, predictions, valid_loss = infer(valid_queue, random_model, criterion, batch_size, num_steps, report_freq, task=task)
    
        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        
        valid_losses.append(valid_loss)
      
        if task == 'next_character_prediction':
            acc = overall_acc(labels, predictions, task)
            logging.info('| epoch {:3d} | valid acc {:5.2f}'.format(epoch, acc))
            valid_acc.append(acc)
    
        else:
            acc = overall_f1(labels, predictions, task)
            logging.info('| epoch {:3d} | valid f1-score {:5.2f}'.format(epoch, acc))
            valid_acc.append(acc)
            
    return train_losses, valid_losses, train_acc, valid_acc, acc
