#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 20:27:50 2021

@author: amadeu
"""

import random
import numpy as np
import time
from generalNAS_tools.genotypes import OPS_cnn, OPS_rnn, PRIMITIVES_cnn, PRIMITIVES_rnn, Genotype



def sample_cnn_operations():
    op = random.choice([1, 2, 3, 4, 5, 6, 7, 8]) # without zero operation
    return op


def sample_rhn_operations():
    op = random.choice([1, 2, 3, 4]) # without zero operation
    return op


def sample_architectures():
    
    ## CNN ##
    cnn_mask = np.zeros([14,9])
    
    # first node receives always two inputs/edges from two previous cells
    
    edge_1_1, edge_1_2 = np.random.choice([2, 3, 4], size=2, replace=False) # sample randomly 2 edges/inputs for each node 2
    
    edge_2_1, edge_2_2 = np.random.choice([5, 6, 7, 8], size=2, replace=False) # sample randomly 2 edges/inputs for each node 3
    
    edge_3_1, edge_3_2 = np.random.choice([9, 10, 11, 12, 13], size=2, replace=False) # sample randomly 2 edges/inputs for each node 4

    # Node1
    cnn_mask[0, sample_cnn_operations()] = 1  
    cnn_mask[1, sample_cnn_operations()] = 1  
    # Node2 
    cnn_mask[edge_1_1, sample_cnn_operations()] = 1  
    cnn_mask[edge_1_2, sample_cnn_operations()] = 1  
    # Node3
    cnn_mask[edge_2_1, sample_cnn_operations()] = 1  
    cnn_mask[edge_2_2, sample_cnn_operations()] = 1  
    # Node4
    cnn_mask[edge_3_1, sample_cnn_operations()] = 1  
    cnn_mask[edge_3_2, sample_cnn_operations()] = 1  


    ## RHN ##
    rnn_mask = np.zeros([36,5])
    edge_1 = random.choice([1, 2])
    edge_2 = random.choice([3, 4, 5])   
    edge_3 = random.choice([6, 7, 8, 9])
    edge_4 = random.choice([10, 11, 12, 13, 14])
    edge_5 = random.choice([15, 16, 17, 18, 19, 20])
    edge_6 = random.choice([21, 22, 23, 24, 25, 26, 27])
    edge_7 = random.choice([28, 29, 30, 31, 32, 33, 34, 35])
    
    # Node0
    rnn_mask[0, sample_rhn_operations()] = 1  
    # Node1
    rnn_mask[edge_1, sample_rhn_operations()] = 1  
    # Node2
    rnn_mask[edge_2, sample_rhn_operations()] = 1  
    # Node3
    rnn_mask[edge_3, sample_rhn_operations()] = 1  
    # Node 4
    rnn_mask[edge_4, sample_rhn_operations()] = 1  
    # Node5
    rnn_mask[edge_5, sample_rhn_operations()] = 1  
    # Node6
    rnn_mask[edge_6, sample_rhn_operations()] = 1  
    # Node7
    rnn_mask[edge_7, sample_rhn_operations()] = 1  
    
    return cnn_mask, rnn_mask


    
def generate_random_architectures(generate_num):
    random_architectures = []
    cnt = 0
    # es werden zuerst random adj und ops matritzen erzeugt und diese
    # werden dann einfach als dictionary eingespeichert und jedes dictionary bildet ein element einer liste archs
    while cnt < generate_num:
        normal_cnn, _ = sample_architectures()
        reduction_cnn, rnn = sample_architectures()

        random_architectures.append([normal_cnn, reduction_cnn, rnn])
        cnt += 1
    return random_architectures



### genotypes
def mask2genotype(random_architecture):
    ## CNN ##
    # normal cell
    cnn_gene = random_architecture[0]
    n = 2
    start = 0
    cell_cnn = []
    for i in range(4):  
        end = start + n
        for j in range(start, end):
            if (cnn_gene[j]==0).all():
               continue
            edge = j-start
            op = PRIMITIVES_cnn[np.nonzero(cnn_gene[j])[0][0]]
            cell_cnn.append((edge,op))
        start = end
        n = n + 1
        
    # reduction cell
    cnn_gene = random_architecture[1]
    n = 2
    start = 0
    cell_cnn_red = []
    for i in range(4):  
        end = start + n
        for j in range(start, end):
            if (cnn_gene[j]==0).all():
               continue
            edge = j-start
            op = PRIMITIVES_cnn[np.nonzero(cnn_gene[j])[0][0]]
            cell_cnn_red.append((edge,op))
        start = end
        n = n + 1
        
    ## RHN ##
    rnn_gene = random_architecture[2]
    cell_rnn = []
    n = 1
    start = 0
    for i in range(8):
        # i =0
        end = start + n
        for j in range(start, end): 
            if (rnn_gene[j]==0).all():
                continue
            edge = j-start
            op = PRIMITIVES_rnn[np.nonzero(rnn_gene[j])[0][0]]
            cell_rnn.append((edge,op))
        start = end
        n = n + 1
    
    genotype = Genotype(
        normal=cell_cnn,
        normal_concat=[2, 3, 4, 5],
        reduce=cell_cnn_red,
        reduce_concat=[2, 3, 4, 5],
        rnn = cell_rnn,
        rnn_concat = range(1,9)
        )
    return genotype