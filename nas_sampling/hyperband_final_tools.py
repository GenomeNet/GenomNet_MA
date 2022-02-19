#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 09:21:29 2021

@author: amadeu
"""


from generalNAS_tools import utils

from randomSearch_and_Hyperband_Tools.hyperbandSampler import create_cnn_supersubnet, create_rhn_supersubnet

#from randomSearch_and_Hyperband_Tools.train_and_validate import train, evaluate_architecture

from generalNAS_tools.train_and_validate_HB import train, infer

from generalNAS_tools.utils import scores_perClass, scores_Overall, pr_aucPerClass, roc_aucPerClass, overall_acc, overall_f1

from randomSearch_and_Hyperband_Tools.hyperbandSampler import maskout_ops, create_new_supersubnet, create_cnn_edge_sampler, create_rhn_edge_sampler

from randomSearch_and_Hyperband_Tools.create_masks import create_cnn_masks, create_rhn_masks

import itertools
import random
import copy

import numpy as np

#logging = logging.getLogger(__name__)
#import logging



# cnn_gene = supernet_mask[0]
def get_final_cnns(cnn_gene):
    cnn_gene = copy.deepcopy(cnn_gene)
    
    cnt=0
    two = None
    for i in range(len(cnn_gene)):
        # i=0
        ops_idxs = np.nonzero(cnn_gene[int(i)])[0]#[0] # gives us the operations, where we can sample from
        if len(ops_idxs) == 2:
            two = True
    
    disc_ops = []
    
    edge_sampler = create_cnn_edge_sampler(cnn_gene) # gives us the edges, where we can sample from

    # random sampling of an edge
    random_cnn_edge = random.choice(edge_sampler)
            
    # random sampling of an operation 
    ops_idxs = np.nonzero(cnn_gene[random_cnn_edge])[0]#[0] # gives us the operations, where we can sample from
    
    random_cnn_op = random.choice(ops_idxs)
            
    disc_ops.append([random_cnn_edge,random_cnn_op])
    criterion = False
    i=1
    while (criterion == False):
        edge_sampler = create_cnn_edge_sampler(cnn_gene) # gives us the edges, where we can sample from

        # random sampling of an edge
        random_cnn_edge = random.choice(edge_sampler)
        
        # random sampling of an operation 
        ops_idxs = np.nonzero(cnn_gene[random_cnn_edge])[0]#[0] # gives us the operations, where we can sample from
        
        random_cnn_op = random.choice(ops_idxs)
        
        #if disc_ops[0] == [random_cnn_edge,random_cnn_op]:
        #    continue
        if [random_cnn_edge,random_cnn_op] in disc_ops[0:i]:
            criterion=False
        else:
            i+=1
            disc_ops.append([random_cnn_edge,random_cnn_op])

            if (two == None) & (i == 3):
                criterion=True
            if (two == True) & (i == 2):
                criterion=True
            
    return disc_ops



def get_final_rhns(rhn_gene):
    
    rhn_gene = copy.deepcopy(rhn_gene)

    two = True
    
    disc_ops = []
    
    edge_sampler = create_rhn_edge_sampler(rhn_gene)
    
    # random sampling of an edge
    random_rhn_edge = random.choice(edge_sampler)
            
    # random sampling of an operation 
    ops_idxs = np.nonzero(rhn_gene[random_rhn_edge])[0]#[0] # gives us the operations, where we can sample from
    
    random_rhn_op = random.choice(ops_idxs)
            
    disc_ops.append([random_rhn_edge,random_rhn_op])
    criterion = False
    i=1
    while (criterion == False):
        edge_sampler = create_rhn_edge_sampler(rhn_gene) # gives us the edges, where we can sample from
    
        # random sampling of an edge
        random_rhn_edge = random.choice(edge_sampler)
        
        # random sampling of an operation 
        ops_idxs = np.nonzero(rhn_gene[random_rhn_edge])[0]#[0] # gives us the operations, where we can sample from
        
        random_rhn_op = random.choice(ops_idxs)
        
        #if disc_ops[0] == [random_cnn_edge,random_cnn_op]:
        #    continue
        if [random_rhn_edge,random_rhn_op] in disc_ops[0:i]:
            criterion=False
        else:
            i+=1
            disc_ops.append([random_rhn_edge,random_rhn_op])
    
            if (two == None) & (i == 3):
                criterion=True
            if (two == True) & (i == 2):
                criterion=True
                
    return disc_ops




def final_stage_run(train_queue, valid_queue, super_model, rhn, conv, criterion, optimizer, epoch, num_steps, clip_params, report_freq, beta, one_clip, task, supernet_mask, budget, batch_size):
    
    
    disc_ops_normal = get_final_cnns(supernet_mask[0])
    disc_ops_reduce = get_final_cnns(supernet_mask[1])
    disc_ops_rhn = get_final_rhns(supernet_mask[2])                
  
    all_discs = [disc_ops_normal,disc_ops_reduce, disc_ops_rhn]
    all_combinations = list(itertools.product(*all_discs))
    
    supernet_mask = copy.deepcopy(supernet_mask)
    
    hyperband_results = []
    
    for discs in all_combinations:
        # discs = all_combinations[0]
        supersubnet_mask = maskout_ops(discs[0], discs[1], discs[2], supernet_mask)
                
        for epoch in range(1):               
           #labels, predictions, valid_loss = infer(valid_queue, supersubnet_model, criterion, args.batch_size, 2*args.num_steps, args.report_freq, task=args.task)
           labels, predictions, valid_loss = infer(valid_queue, super_model, criterion, batch_size, 2*num_steps, report_freq, task, supersubnet_mask)
           #logging.info('| valid loss{:5.2f}'.format(valid_loss))
           
           labels = np.concatenate(labels)
           predictions = np.concatenate(predictions)
                                    
           if task == 'next_character_prediction':
               acc = overall_acc(labels, predictions, task)
               #logging.info('| epoch {:3d} | valid acc {:5.2f}'.format(epoch, acc))
               #valid_acc.append(acc)
           else:
               acc = overall_f1(labels, predictions, task)
               # logging.info('| epoch {:3d} | valid f1-score {:5.2f}'.format(epoch, acc))
               #valid_acc.append(acc)
           # valid_acc, valid_obj = evaluate_architecture(valid_object, supersubnet_model, criterion, optimizer, epoch) 
           hyperband_results.append([supersubnet_mask, valid_loss]) # hätt

    def acc_position(list):
        return list[1]
    
    hyperband_results.sort(reverse=False, key=acc_position) # in descending order: best/highest acc is on top (contains the worst operations which we want to discard)
    # here, we use ascending order: best/lowest val_loss is on top (contains the worst operations which we want to discard)
    
    # discard the edges and operations from the good performing supersubnets to build new S
    # keep mask with best/highest acc 
    # supernet_mask = create_new_supersubnet(hyperband_results, supernet_mask)
    supernet_mask = hyperband_results[0][0]
    
    return supernet_mask, hyperband_results
        
        
              
            #supersubnet_mask = maskout_ops(disc_ops_normal[0], disc_ops_reduce[0], disc_ops_rhn[0], supernet_mask)
            
            #supersubnet_mask = maskout_ops(disc_ops_normal[0], disc_ops_reduce[0], disc_ops_rhn[1], supernet_mask)
            #supersubnet_mask = maskout_ops(disc_ops_normal[0], disc_ops_reduce[1], disc_ops_rhn[0], supernet_mask)
            #supersubnet_mask = maskout_ops(disc_ops_normal[1], disc_ops_reduce[0], disc_ops_rhn[0], supernet_mask)
            
            #supersubnet_mask = maskout_ops(disc_ops_normal[0], disc_ops_reduce[1], disc_ops_rhn[1], supernet_mask)
            #supersubnet_mask = maskout_ops(disc_ops_normal[1], disc_ops_reduce[1], disc_ops_rhn[0], supernet_mask)
            #supersubnet_mask = maskout_ops(disc_ops_normal[1], disc_ops_reduce[0], disc_ops_rhn[1], supernet_mask)
            
            #supersubnet_mask = maskout_ops(disc_ops_normal[1], disc_ops_reduce[1], disc_ops_rhn[1], supernet_mask)
    
    
            