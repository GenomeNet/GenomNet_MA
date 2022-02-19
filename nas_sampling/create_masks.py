#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 14:03:14 2021

@author: amadeu
"""

import random
import numpy as np
import time
from generalNAS_tools.genotypes import OPS_cnn, OPS_rnn, PRIMITIVES_cnn, PRIMITIVES_rnn, Genotype

import copy

    
    

# cnn_edge = cnn_mask[0]
def sample_cnn_operations(cnn_edge):
    op_idx = np.where(cnn_edge == 0)[0]
    op = random.choice(op_idx) # without zero operation
    return op

# cnn_nodes = cnn_mask[0:2]
def sample_cnn_edges(cnn_nodes, offset):
    edge_idx = []
    for i, edge in enumerate(cnn_nodes):
        if (edge!=0).all():
            continue
        
        edge_idx.append(i+offset)
        
    edge_1 = random.choice(edge_idx) # without zero operation
    edge_2 = random.choice(edge_idx) # without zero operation

    return edge_1, edge_2


def sample_rhn_operations():
    op = random.choice([1, 2, 3, 4]) # without zero operation
    return op


def sample_architectures(cnn_mask):
    
    ## CNN ##
    
    # Edges Node1    
    edge_0_1, edge_0_2 = sample_cnn_edges(cnn_mask[0:2], 0)
    
    # Edges Node2 
    edge_1_1, edge_1_2 = sample_cnn_edges(cnn_mask[2:5], 2)
    
    # Edges Node3 
    edge_2_1, edge_2_2 = sample_cnn_edges(cnn_mask[5:9], 5)
    
    # Edges Node3 
    edge_3_1, edge_3_2 = sample_cnn_edges(cnn_mask[9:14], 9)


    # Node1
    cnn_mask[edge_0_1, sample_cnn_operations(cnn_mask[edge_0_1])] = 1  
    cnn_mask[edge_0_2, sample_cnn_operations(cnn_mask[edge_0_2])] = 1 
    
    # Node2 
    cnn_mask[edge_1_1, sample_cnn_operations(cnn_mask[edge_1_1])] = 1 
    cnn_mask[edge_1_2, sample_cnn_operations(cnn_mask[edge_1_2])] = 1  
    # Node3
    cnn_mask[edge_2_1, sample_cnn_operations(cnn_mask[edge_2_1])] = 1  
    cnn_mask[edge_2_2, sample_cnn_operations(cnn_mask[edge_2_2])] = 1  
    # Node4
    cnn_mask[edge_3_1, sample_cnn_operations(cnn_mask[edge_3_1])] = 1  
    cnn_mask[edge_3_2, sample_cnn_operations(cnn_mask[edge_3_2])] = 1  

    return cnn_mask




# CNN
def create_cnn_sampler(cnn_gene):
    # cnn_gene = supernet_mask[0]
    n = 2
    start = 0
    #edge_sampler = np.empty((0,1), int)
    #remain_cnn_ops = []
    sampler=[]
    for i in range(4):  
        # i = 0
        end = start + n
        #cnt = 0
        for j in range(start, end):
            # j =1
            possible_ops = np.where(cnn_gene[j] == 0)[0] # possible elements
            possible_ops = possible_ops[1::]

            if possible_ops.size >= 1:
               # cnt += 1
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
            # i=0
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
        # i = 0
        end = start + n
        # masks = []
        # cnt = 0
        for j in range(start, end): 
            # j =0
            possible_ops = np.where(rhn_gene[j] == 0)[0] # possible elements
            possible_ops = possible_ops[1::]


            if possible_ops.size >= 1:
               # cnt += 1
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






    

    
def create_cnn_supersubnet(supernet_mask, num_disc):
    cnn_gene = copy.deepcopy(supernet_mask)
    selected_ops = []

    for i in range(num_disc):
        edge_sampler = create_cnn_edge_sampler(cnn_gene) # gives us the edges, where we can sample from
        if edge_sampler.size != 0:

            # random sampling of an edge
            random_cnn_edge = random.choice(edge_sampler)
            
            # random sampling of an operation 
            ops_idxs = np.nonzero(cnn_gene[random_cnn_edge])[0]#[0] # gives us the operations, where we can sample from
            random_cnn_op = random.choice(ops_idxs)
            
            # discard corresponding operation to build new supersubnet
            cnn_gene[random_cnn_edge, random_cnn_op] = 0 
            disc_ops.append([random_cnn_edge,random_cnn_op])
            
    return cnn_gene, disc_ops

























