#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:52:01 2021

@author: amadeu
"""


import copy

from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype

import numpy as np


def parse_network(switches_normal_cnn, switches_reduce_cnn, switches_rnn):

    def _parse_switches_cnn(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            # i=1
            end = start + n
            for j in range(start, end):
                # j=0
                for k in range(len(switches[j])): 
                    if switches[j][k]:
                        gene.append((PRIMITIVES_cnn[k], j - start)) 
            start = end
            n = n + 1
        return gene
    
    gene_normal = _parse_switches_cnn(switches_normal_cnn)
    gene_reduce = _parse_switches_cnn(switches_reduce_cnn)
    
    concat = range(2, 6)
    
    #genotype_cnn = Genotype_cnn(
    #    normal=gene_normal, normal_concat=concat, 
    #    reduce=gene_reduce, reduce_concat=concat
    #)
    
    def _parse_switches_rnn(switches):
        n = 1
        start = 0
        gene = []
        step = 8
        for i in range(step):
            
            end = start + n
            for j in range(start, end): 
             
                for k in range(len(switches[j])): 
                    if switches[j][k]:
                        gene.append((PRIMITIVES_rnn[k], j - start)) 
            start = end
            n = n + 1
        return gene
    
    gene_rnn = _parse_switches_rnn(switches_rnn)
                
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat,
        rnn = gene_rnn, rnn_concat = range(rnn_steps+1)[-CONCAT:]
    )
            
    return genotype


def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input) 
        index.append(idx)
        input[idx] = 1
    return index


def get_max_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmax(input) 
        index.append(idx)
        input[idx] = -1
    return index

# drop = get_min_k_no_zero(normal_prob[i-switch_count, :], idxs, num_to_drop[sp])
# w_in, idxs, k = normal_prob[i-switch_count, :], idxs, 2
def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True 
    else:
        zf = False
    if zf:
        w = w[1:] 
        index.append(0) 
        k = k - 1 
    for i in range(k): 
        # i=0
        idx = np.argmin(w) 
        w[idx] = 1 
        if zf:
            idx = idx + 1
        index.append(idx) 
    return index

        
def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES_cnn[j])
        logging.info(ops)
   
# switches_normal_cnn
def check_sk_number(switches):
    count = 0
    for i in range(len(switches)): 
        # i=0
        if switches[i][3]: # element 3 stands for 'identity' operation
            count = count + 1
    
    return count


# switches_in, switches_bk, probs_in = switches_normal_cnn, switches_normal_2, normal_prob
# switches_normal_cnn, switches_normal_2, normal_prob
def delete_min_sk_prob(switches_in, switches_bk, probs_in): 
    
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]: # if element scipconnect not True -> means False, then idx=-1
            idx = -1
        else: # if this row has scipconnect (True) -> return idx, which gives us the corresponding index_element of the scip connection in the normal_prob matrix
            idx = 0
            for i in range(3): 
            # 
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx
    
    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))] # 14 elements with values with 1.0
    disc_edg_cnt = 0
    for i in range(len(switches_in)): 
        # i=8
        if switches_bk[i].count(False) != len(switches_bk[i]):
            idx = _get_sk_idx(switches_in, switches_bk, i)  
            if not idx == -1: # then there is scip connect
                sk_prob[i] = probs_out[i-disc_edg_cnt][idx] # define sk_prob element with the corresponding scip connect alpha value
        else:
            disc_edg_cnt += 1
            
    d_idx = np.argmin(sk_prob) # d_idx is the index of the lowest scip connect row/edge
    idx = _get_sk_idx(switches_in, switches_bk, d_idx) # gives us the scipconnect column of this edge
    
    # get number of discarded edges before the scip connect edge
    disc_edg_cnt = 0
    for i in range(d_idx):
        if switches_bk[i].count(False) == len(switches_in[i]):
            disc_edg_cnt += 1
    
    probs_out[d_idx-disc_edg_cnt][idx] = 0.0 # set the corresponding element to 0 in normal_prob 
    
    return probs_out


# switches_in, probs = switches_normal_2, normal_prob
def keep_1_on(switches_in, probs): # in order to have only one operation per edge

    switches = copy.deepcopy(switches_in)
    disc_edg_cnt = 0
    for i in range(len(switches)):
        if switches[i].count(False) != len(switches[i]):
            # i=7
            idxs = []
            for j in range(len(PRIMITIVES_cnn)):
                if switches[i][j]:
                    idxs.append(j)
            drop = get_min_k_no_zero(probs[i-disc_edg_cnt, :], idxs, 2) 
            for idx in drop:
                switches[i][idxs[idx]] = False     
        else:
            disc_edg_cnt += 1
            
    return switches



# switches_in, probs = switches_normal_cnn, normal_prob
def keep_2_branches(switches_in, probs): # in order to have only 2 edges per node
    switches = copy.deepcopy(switches_in)
    
    final_prob = [0.0 for i in range(len(switches))]
    
    disc_edg_cnt = 0
    for i in range(len(switches)): 
        if switches[i].count(False) != len(switches[i]):
            final_prob[i] = max(probs[i-disc_edg_cnt]) # add best/max operation to final_prob 
        else:
            final_prob[i] = 0
            disc_edg_cnt += 1
         
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3): 
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
        
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES_cnn)):
                switches[i][j] = False  
    return switches  