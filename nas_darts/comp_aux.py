#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:38:29 2021

@author: amadeu
"""


# The aim of this script is to make the RHN part faster than the original DARTS implementation

import torch

import numpy as np

from generalNAS_tools.operations_14_9 import *
from generalNAS_tools.genotypes import PRIMITIVES_rnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# auxiliary functions to make CNN part faster and enable continuous weight sharing as well as discarded edges


def get_state_ind(cnn_steps, switch_cnn):
    start = 0
    n = 2
    idxs = []
    
    for i in range(cnn_steps):
        
        end = start + n
        idx = []
        
        for j in range(start, end):
            if switch_cnn[j].count(False) != len(switch_cnn[j]):
                idx.append(j-start)
                   
        idxs.append(torch.tensor(idx).to(device))
        start = end
        
        n += 1
        
    return idxs


            

# cnn_steps, switch_cnn = _steps, switches
def get_w_pos(cnn_steps, switch_cnn):
    
    start = 0
    n = 2
    idxs = [0]
    idx = 0
    
    for i in range(cnn_steps-1):
        
        end = start + n
        
        for j in range(start, end):
            if switch_cnn[j].count(False) != len(switch_cnn[j]):
                idx += 1
     
                
        idxs.append(idx)
        start = end
        
        n += 1
        
    return idxs



def get_w_range(cnn_steps, switch_cnn):
    
    start = 0
    n = 2
    idxs = []
    idx = 0
    
    for i in range(cnn_steps):
        # i=1
        end = start + n
        
        for j in range(start, end):
            if switch_cnn[j].count(False) != len(switch_cnn[j]):
                idx += 1
     
        if i < 1:
            idxs.append([0, idx])
        else:   
            idxs.append([idxs[i-1][1],idx])
        start = end
        
        n += 1
        
    return idxs




# auxiliary functions to make RHN part faster and enable continuous weight sharing as well as discarded edges

def get_disc_edges(rnn_steps, switch_rnn):
    offset = 0
    disc_edge = []
    disc_cnt = 0
    
    for i in range(rnn_steps):
     
            
        for j in range(offset, offset+i+1):
           
            if switch_rnn[j].count(False) == len(switch_rnn[j]):
                disc_cnt += 1 
            
            disc_edge.append(disc_cnt)
        
        offset += i+1
        
    return disc_edge






def activ_fun(rnn_steps, PRIMITIVES_rnn, switch_rnn):
    #
    act_all = []
    act_nums = []
    
    offset = 0
    
    for i in range(rnn_steps):
        # i=0
        
        switch_node = np.array(switch_rnn[offset:offset+i+1])
        act_step = []
        act_num = []

        for k, name in enumerate(PRIMITIVES_rnn): # geht von 0:4
        
            if name == 'none':
                continue
            
            if list(switch_node[:,k]).count(False) != len(switch_node[:,k]):
                act_step.append(name)
                act_num.append(k)
                
        offset += i+1     
        act_all.append(act_step)
        act_nums.append(act_num)

        
    return act_all, act_nums

          

def compute_positions(switch_rnn, rnn_steps, PRIMITIVES_rnn):
    
    acn, acf = activ_fun(rnn_steps, PRIMITIVES_rnn, switch_rnn)  
    
    disc_edge = get_disc_edges(rnn_steps, switch_rnn)


    positions = []

    offset = 0
    rows = []
    cols = []
    nodes = []
    
    for i in range(len(acf)):
        
        for k in acf[i]:
            
            row = []
            col = []
            nod = []
            
            for j in range(offset, offset+i+1):
                # j: 0 (offset=0)
                # j: 1,2 (offset=1)
                # j: 3,4,5 (offset=3)
                # j: 6,7,8,9
                if switch_rnn[j][k]:
                    cnt_true = switch_rnn[j][0:k].count(True)
                    row.append(j-disc_edge[j])
                    nod.append(j-offset)
                    col.append(cnt_true) # 
                    
            rows.append(torch.tensor(row).to(device))
            nodes.append(torch.tensor(nod).to(device))
            cols.append(torch.tensor(col).to(device))
            
            
        offset += i+1
    return rows, nodes, cols



