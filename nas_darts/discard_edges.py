#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:15:38 2021

@author: amadeu
"""


from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn
import torch.nn.functional as F

import torch

import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



## discard CNN edges ##

def discard_cnn_edges(model, switches_normal_cnn, switches_reduce_cnn, max_edges_cnn, sp):
    
    switches_normal_cnn = copy.deepcopy(switches_normal_cnn)
    switches_reduce_cnn = copy.deepcopy(switches_reduce_cnn)
    
    nor_arch = F.softmax(model.alphas_normal, dim=-1).data.cpu().numpy()
    red_arch = F.softmax(model.alphas_reduce, dim=-1).data.cpu().numpy()
      
    final_prob_normal = [0.0 for i in range(len(switches_normal_cnn))]
    final_prob_reduce = [0.0 for i in range(len(switches_reduce_cnn))]
    
    switch_count_normal = 0
    switch_count_reduce = 0
    
    for i in range(len(switches_normal_cnn)): 
        if switches_normal_cnn[i].count(False) != len(switches_normal_cnn[i]): # if they are not all false -> this is not a discarded edge
            if switches_normal_cnn[i][0] ==True:
                nor_arch[i-switch_count_normal][0] = 0
            final_prob_normal[i] = max(nor_arch[i-switch_count_normal]) # add best/max operation to final_prob 
        else: # if discarded edge
            final_prob_normal[i] = 1 # set previous discarded edge to 1 (so that it does not get discarded in this stage)
            switch_count_normal += 1
            
        if switches_reduce_cnn[i].count(False) != len(switches_reduce_cnn[i]): # if no discarded edge
            if switches_reduce_cnn[i][0] ==True:
                red_arch[i-switch_count_reduce][0]=0
            final_prob_reduce[i] = max(red_arch[i-switch_count_reduce])
        else: # if discarded edge
            final_prob_reduce[i] = 1
            switch_count_reduce += 1
        
          
    discard_switch_normal = [] 
    discard_switch_reduce = []
    max_edge = max_edges_cnn[sp+1]
    n = 3
    start = 2
    
    for i in range(3):
        # i=0
        # print(i): 0,1,2
        # which_node: with sp4 only 3rd, with sp5 2nd and 3rd, with sp6 1st, 2nd and 3rd
        # 4-3=1; 
        # 4-1=3, 4-2=2, 4-3=1
        # 5-1=4, 5-2=3, 5-3=2
        # i =2
        end = start + n
        
        edges_normal = final_prob_normal[start:end]
        edges_reduce = final_prob_reduce[start:end]
        
        disc_edg = 0
        for j in edges_normal:
            if j == 1:
                disc_edg+=1
        num_edges = len(edges_normal) - disc_edg
        
        num_disc = num_edges - max_edge
            
        if num_edges > max_edge:
            edge_n = sorted(range(n), key=lambda x: edges_normal[x]) # sort by this, from worst to best
            discard_edge_n = edge_n[0:num_disc] # worst of all is 0th element, therefore edge[0]
            for j in discard_edge_n:
                discard_switch_normal.append(j+start)
            
            edge_r = sorted(range(n), key=lambda x: edges_reduce[x]) # sort by this, from worst to best
            discard_edge_r = edge_r[0:num_disc] # worst of all is 0th element, therefore edge[0]
            for j in discard_edge_r:
                discard_switch_reduce.append(j+start)
            
        
        start = end
        n = n + 1
    
    num_discard_n, num_discard_r = 0, 0
    
    switch_count_normal = 0
    switch_count_reduce = 0
    
    for i in range(1,len(switches_normal_cnn)):
        # i=0
        if i in discard_switch_normal: # if an edge should be discarded in this stage
            
            num_discard_n += 1
                           
            for j in range(len(PRIMITIVES_cnn)):
                switches_normal_cnn[i][j] = False 
        
        
        if i in discard_switch_reduce: 
            
            num_discard_r += 1
            
            for j in range(len(PRIMITIVES_cnn)):
                switches_reduce_cnn[i][j] = False  
            
    return switches_normal_cnn, switches_reduce_cnn




def discard_rhn_edges(model, switches_rnn, max_edges_rhn, sp):
    
    switches_rnn = copy.deepcopy(switches_rnn)

    rnn_final = [0 for idx in range(36)]

    arch_rhn = F.softmax(model.weights, dim=-1).data.cpu().numpy()

    switch_count_rnn = 0

    for i in range(len(switches_rnn)): 
        # i=0
        if switches_rnn[i].count(False) != len(switches_rnn[i]): # if not all false -> this is not a discarded edge
            if switches_rnn[i][0] == True: # set zero operation to zero
                arch_rhn[i-switch_count_rnn][0] = 0
            
            rnn_final[i] = max(arch_rhn[i-switch_count_rnn]) # add best/max operation to final_prob 
        else: # if previous discarded edge
            rnn_final[i] = 1 # set previous discarded edge to 1 (so that it does not get discarded in this stage)
            switch_count_rnn += 1
    
        
    start = 1
    discard_switch_rnn=[]
    max_edge = max_edges_rhn[sp+1]
    n=2
    for i in range(7):
         # i=0
         end = start + i + 2 
                          
         tbrnn = rnn_final[start:end]
         
         disc_edg = 0
         for j in tbrnn:
             if j == 1:
                 disc_edg+=1
         num_edges = len(tbrnn) - disc_edg
         num_disc = num_edges - max_edge

         if num_edges > max_edge:
             # with sp=1 in last iter therefore i=6 discard between [28,35]
             # with sp=2 in last two iters therefore i=5 and i=6 so discard between [21,27] and [28,35] one edge each
         
             edge_rnn = sorted(range(n), key=lambda x: tbrnn[x]) 

             
             discard_edge_rnn = edge_rnn[0:num_disc] # worst of all is 0th, therefore edge[0]
             for j in discard_edge_rnn:
                 discard_switch_rnn.append(j+start)
    
         start = end
         n +=1
                 
    
    num_discard = 0
    switch_count_rnn = 0
    for i in range(1, len(switches_rnn)):
        if i in discard_switch_rnn: # if an edge should be discarded in this stage
            
            num_discard += 1
                                           
            for j in range(len(PRIMITIVES_rnn)):
                switches_rnn[i][j] = False
                
    return switches_rnn
    
