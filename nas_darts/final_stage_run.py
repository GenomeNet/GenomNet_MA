#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:18:12 2021

@author: amadeu
"""


from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype
import torch.nn.functional as F

from darts_tools.auxiliary_functions import parse_network

import numpy




def final_stage_genotype(model, switches_normal_cnn, switches_normal_2, switches_reduce_cnn, switches_reduce_2, switches_rnn, switches_rnn2):
    
    arch_param = model.arch_parameters() # model.module.arch_parameters()
            
    sm_dim=-1
    
    normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
    reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
    rnn_prob = F.softmax(arch_param[2], dim=sm_dim).data.cpu().numpy()
    
    # s_nr = switches_normal_cnn
    # s_rr = switches_reduce_cnn
    # s_rnnr = switches_rnn
    # switches_reduce_2
    # switches_rnn, switches_normal_cnn, switches_reduce_cnn = s_rnnr, s_nr, s_rr
    
    normal_final = [0 for idx in range(14)] 
    reduce_final = [0 for idx in range(14)]
    rnn_final = [0 for idx in range(36)]
      
    switch_count_cnn = 0
    for i in range(14):
        if switches_normal_cnn[i].count(False) != len(switches_normal_cnn[i]):
            # i=0
            if switches_normal_2[i][0] == True: # ensure that 'none' is nether in normal cell of final architecture 
                normal_prob[i-switch_count_cnn][0] = 0
            normal_final[i] = max(normal_prob[i-switch_count_cnn]) 
        else:
            switch_count_cnn += 1
            normal_final[i] = 0
        
    switch_count_cnn = 0
    for i in range(14):
        if switches_reduce_cnn[i].count(False) != len(switches_reduce_cnn[i]):
            if switches_reduce_2[i][0] == True:  # ensure that 'none' is nether in reduction cell of final architecture
                reduce_prob[i-switch_count_cnn][0] = 0
            reduce_final[i] = max(reduce_prob[i-switch_count_cnn])  
        else:
            switch_count_cnn += 1
            reduce_final[i] = 0
        
    switch_count_rnn = 0
    for i in range(36):
        if switches_rnn[i].count(False) != len(switches_rnn[i]):
            if switches_rnn2[i][0] == True: # ensure that 'none' is nether in final architecture
                rnn_prob[i-switch_count_rnn][0] = 0
            rnn_final[i] = max(rnn_prob[i-switch_count_rnn])
        else:
            switch_count_rnn += 1
            rnn_final[i] = 0
    
    
    # Generate Architecture, similar to DARTS
    keep_normal = [0, 1] 
    keep_reduce = [0, 1]
    n = 3
    start = 2
    
    for i in range(3):
        # i=
        end = start + n
          
        tbsn = normal_final[start:end] 
        tbsr = reduce_final[start:end]
        
        edge_n = sorted(range(n), key=lambda x: tbsn[x]) 
        keep_normal.append(edge_n[-1] + start) 
        keep_normal.append(edge_n[-2] + start)
        
        edge_r = sorted(range(n), key=lambda x: tbsr[x])
        keep_reduce.append(edge_r[-1] + start)
        keep_reduce.append(edge_r[-2] + start)
        
        start = end
        n = n + 1                  
        
        
    keep_rnn = [0]
    start = 1
    n=2
    
    for i in range(7):
    
        end = start + i + 2 
        
        tbrnn = rnn_final[start:end]
        
        edge_rnn = sorted(range(n), key=lambda x: tbrnn[x]) 
       
        keep_rnn.append(edge_rnn[-1] + start)
       
        start = end
        n +=1
    
    
    for i in range(14):
        if not i in keep_normal: # if an edge is not in the final_architecture -> set all values to False (normal cell)
            
            for j in range(len(PRIMITIVES_cnn)):
                switches_normal_cnn[i][j] = False
                
        if not i in keep_reduce: # if an edge is not in the final_architecture -> set all values to False (reduction cell)
            for j in range(len(PRIMITIVES_cnn)):
                switches_reduce_cnn[i][j] = False
            
    for i in range(36):
        if not i in keep_rnn: # if an edge is not in the final_architecture -> set all values to False
            for j in range(len(PRIMITIVES_rnn)):
                switches_rnn[i][j] = False
            
    # translate switches into genotype
    genotype = parse_network(switches_normal_cnn, switches_reduce_cnn, switches_rnn)
    
    return genotype, switches_normal_cnn, switches_reduce_cnn, switches_rnn, normal_prob, reduce_prob, rnn_prob



