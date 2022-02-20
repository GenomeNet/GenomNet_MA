#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:09:30 2021

@author: amadeu
"""


from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype
import torch.nn.functional as F

import torch

from darts_tools.auxiliary_functions import *

import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model, switches_normal_cnn, switches_reduce_cnn, num_to_keep, num_to_drop, sp, new_alpha_values=False



def discard_cnn_ops(model, switches_normal_cnn, switches_reduce_cnn, num_to_keep, num_to_drop, sp, new_alpha_values=True):

        
    arch_normal = model.alphas_normal # ist immer auf GPU
    normal_prob = F.softmax(arch_normal, dim=-1).data.cpu().numpy()     
    # normal_prob = F.softmax(alphas_normal, dim=-1).data.cpu().numpy()     
    # update switches from CNN
    new_normal_arch = torch.empty(1,num_to_keep[sp])#.cpu()
    
    switch_count = 0
    
     # arch_param = model.arch_parameters()
     # normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()     
    
    for i in range(14): # for each connection/edge
        # i=0
        if switches_normal_cnn[i].count(False) != len(switches_normal_cnn[i]): # if all values in row are false -> discarded edge
    
            # i = 0
            idxs = [] 
            for j in range(len(PRIMITIVES_cnn)): # for each operation
                if switches_normal_cnn[i][j]:
                    idxs.append(j) # nur wenn True ist wird gezählt, also ist idxs: 9,8,7,6,...,0
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(normal_prob[i-switch_count, :], idxs, num_to_drop[sp])
            else: 
                drop = get_min_k(normal_prob[i-switch_count, :], num_to_drop[sp])
                
                keep = np.sort(get_max_k(normal_prob[i-switch_count, :], num_to_keep[sp]))
                ## reinitialize the one-shot model of next stage with following alpha values
                # chose the worst performing operation
                if new_alpha_values==True:
                    # dropped_alpha = arch_normal[i-switch_count, drop]    
                    # discard the worst performing operation 
                    new_row = arch_normal[i,keep].reshape(1,num_to_keep[sp])
                    #new_row = arch_normal[i-switch_count,:][arch_normal[i-switch_count,:]!=dropped_alpha].view(1,num_to_keep[sp]) # müsste immmer auf GPU sein
                    
                    new_normal_arch = torch.cat((new_normal_arch.to(device), new_row)) # ich glaube new_normal_arch ist eben auf CPU und das muss geändert werden!!
                
            for idx in drop: # 3 dann 1
                switches_normal_cnn[i][idxs[idx]] = False 
        else: # if discarded edge
            switch_count += 1
            
    new_normal_arch = new_normal_arch[1:15-switch_count,:]
     
    arch_reduce = model.alphas_reduce
    reduce_prob = F.softmax(arch_reduce, dim=-1).data.cpu().numpy()
    new_reduce_arch = torch.empty(1,num_to_keep[sp])
    switch_count = 0

    for i in range(14):
        # i=0
        
        if switches_reduce_cnn[i].count(False) != len(switches_reduce_cnn[i]):

            # i = 0
            idxs = []
            for j in range(len(PRIMITIVES_cnn)):
                if switches_reduce_cnn[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(reduce_prob[i-switch_count, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(reduce_prob[i-switch_count, :], num_to_drop[sp])
                keep = np.sort(get_max_k(reduce_prob[i-switch_count, :], num_to_keep[sp]))

                
                if new_alpha_values==True:
                    # dropped_alpha = arch_reduce[i-switch_count, drop]    
                    # discard the worst performing operation
                    # new_row = arch_reduce[i-switch_count,:][arch_reduce[i-switch_count,:]!=dropped_alpha].view(1,num_to_keep[sp])
                    new_row = arch_reduce[i,keep].reshape(1,num_to_keep[sp])
                    
                    new_reduce_arch = torch.cat((new_reduce_arch.to(device), new_row))
                
            for idx in drop:
                
                switches_reduce_cnn[i][idxs[idx]] = False
        else:
            switch_count +=1
    
            
    new_reduce_arch = new_reduce_arch[1:15-switch_count,:] 
    
    return new_normal_arch, new_reduce_arch, switches_normal_cnn, switches_reduce_cnn






def discard_rhn_ops(model, switches_rnn, num_to_keep_rnn, num_to_drop_rnn, sp, new_alpha_values=True):

    arch_rnn = model.weights
    rnn_prob = F.softmax(arch_rnn, dim=-1).data.cpu().numpy()
    new_arch_rnn = torch.empty(1, num_to_keep_rnn[sp])
    switch_count = 0
    for i in range(36):
        # i=23
        if switches_rnn[i].count(False) != len(switches_rnn[i]):
            idxs = []
            for j in range(len(PRIMITIVES_rnn)):
                # j=0
                if switches_rnn[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep_rnn) - 1:
                drop = get_min_k_no_zero(rnn_prob[i-switch_count, :], idxs, num_to_drop_rnn[sp])
               
            else:
                drop = get_min_k(rnn_prob[i-switch_count, :], num_to_drop_rnn[sp])
                keep = np.sort(get_max_k(rnn_prob[i-switch_count, :], num_to_keep_rnn[sp]))

                if new_alpha_values==True:
                    # dropped_alpha = arch_rnn[i-switch_count, drop]    
                    new_row = arch_rnn[i,keep].reshape(1,num_to_keep_rnn[sp])
                    # discard the worst performing operation from alpha-matrix
                    #new_row = arch_rnn[i-switch_count,:][arch_rnn[i-switch_count,:]!=dropped_alpha].view(1,num_to_keep_rnn[sp])
                        
                    new_arch_rnn = torch.cat((new_arch_rnn.to(device), new_row))
            for idx in drop:
                switches_rnn[i][idxs[idx]] = False
        else:
            switch_count +=1
            
            
    new_arch_rnn = new_arch_rnn[1:37-switch_count,:]
    
    return new_arch_rnn, switches_rnn