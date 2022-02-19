#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 14:03:14 2021

@author: amadeu
"""

##################################################################
#### used by OSP-NAS



import random
import numpy as np

# CNN
def create_cnn_sampler(cnn_gene):
    n = 2
    start = 0
    sampler=[]
    for i in range(4):  
        end = start + n
        for j in range(start, end):
            possible_ops = np.where(cnn_gene[j] == 0)[0] # possible elements
            possible_ops = possible_ops[1::]

            if possible_ops.size >= 1:
                sampler.append((j, possible_ops))
                
        start = end
        n = n + 1
    return sampler


def create_cnn_masks(initial_num):
    
    cnn_gene = np.zeros([14,9])

    cnt_cur=0
    
    sampler = create_cnn_sampler(cnn_gene)
    
    while cnt_cur < initial_num:
        sample = random.choice(sampler)
        edge_idx = sample[0]
        op_idx = random.choice(sample[1])    
            
        cnn_gene[edge_idx, op_idx] = 1.0
        cnt_cur=0

        for i in range(len(cnn_gene)):
            for j in range(len(cnn_gene[i])):
                if cnn_gene[i][j]!=0:
                    cnt_cur += 1
    return cnn_gene
        
    
# RHN
def create_rhn_sampler(rhn_gene):
    
    n = 1
    start = 0
    sampler = []  
    
    for i in range(8):
        end = start + n
        for j in range(start, end): 
            possible_ops = np.where(rhn_gene[j] == 0)[0] # possible elements
            possible_ops = possible_ops[1::]


            if possible_ops.size >= 1:
                sampler.append((j, possible_ops))
    
        start = end
        n = n + 1
        
    return sampler



def create_rhn_masks(initial_num):
    
    rhn_gene = np.zeros([36,5])

    cnt_cur=0
    
    sampler = create_rhn_sampler(rhn_gene)


    while cnt_cur < initial_num:
        sample = random.choice(sampler)
        edge_idx = sample[0]
        op_idx = random.choice(sample[1])    
            
        rhn_gene[edge_idx, op_idx] = 1.0
        cnt_cur=0

        for i in range(len(rhn_gene)):
            # i=0
            for j in range(len(rhn_gene[i])):
                if rhn_gene[i][j]!=0:
                    cnt_cur += 1
    return rhn_gene
