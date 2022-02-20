#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:38:29 2021

@author: amadeu
"""


# The aim of this script, is to make the RHN part faster than the original DARTS implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
#from genotypes_rnn import PRIMITIVES, STEPS, CONCAT, total_genotype
from torch.autograd import Variable
from collections import namedtuple
# from model import DARTSCell, RNNModel
# from darts_tools.model import DARTSCell, RNNModel

import numpy as np

from generalNAS_tools.operations_14_9 import *
#from genotypes_cnn import PRIMITIVES_cnn
#from genotypes_cnn import Genotype_cnn
from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from darts_tools.genotype_parser import parse_genotype





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




#idxs[0][0]
#alphas_normal[0:2]

#alphas_normal[idxs[i][0]:idxs[i][1]]


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







# switch_rnn = switches_rnn
# switch_rnn[25][0] = True
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

          
      
#acn, acf = activ_fun(rnn_steps, PRIMITIVES_rnn, switch_rnn) 




def compute_positions(switch_rnn, rnn_steps, PRIMITIVES_rnn):
    
    acn, acf = activ_fun(rnn_steps, PRIMITIVES_rnn, switch_rnn)  
    
    disc_edge = get_disc_edges(rnn_steps, switch_rnn)


    positions = []
    # step_position = []
    offset = 0
    rows = []
    cols = []
    nodes = []
    
    for i in range(len(acf)):
        # i=3
        
        for k in acf[i]:
            # k=4
            
            row = []
            col = []
            nod = []
            
            for j in range(offset, offset+i+1):
                # j=9
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






# rows, nodes, cols = compute_positions(switch_rnn, rnn_steps)


#x_mask = h_mask = x = h_prev = torch.rand(2,256)
#probs = new_rnn_arch
#s0 = torch.rand(2,256)
# states_original = states
#cnt = 0     
#offset = 0
#states = s0.unsqueeze(0) 
#disc_edge_count = 0

#for i in range(rnn_steps):
    # i=7
#    masked_states = states
 
#    ch = masked_states.view(-1, nhid).mm(_Ws[i]).view(i+1, -1, 2*nhid) 
    
#    c, h = torch.split(ch, nhid, dim=-1)
       
#    c = c.sigmoid()
    
#    s = torch.zeros_like(s0) # [2,256]
    #s = s.unsqueeze(0)
    #unweighteds = [states + c * (torch.tanh(h) - states), states + c * (torch.sigmoid(h) - states), states + c * (torch.relu(h) - states), states + c * (h - states)]

#    for name in acn[i]: # geht von 0:4
        # k=4
        #print(name)
        
        # name = PRIMITIVES_rnn[k]
        
        #if name == 'none':
        #  continue
#        fn = _get_activation(name)
        # fn = _get_activation(name)

#        unweighted = states + c * (fn(h) - states) # states [3,2,256], wobei s immer [2,256]
          
#        s += torch.sum(probs[rows[cnt], cols[cnt]].unsqueeze(-1).unsqueeze(-1) * unweighted[nodes[cnt], :, :], dim=0) 
        # sm = torch.sum(probs[rows[cnt], cols[cnt]].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0) 
        
#        cnt += 1 

#    states = torch.cat([states, s.unsqueeze(0)], 0) # von [1,2,256] zu [2,2,256]



#w = torch.rand(3,1,1)
#w[0] = 1
#A = torch.rand(3,2,4)
#sm = torch.sum(w[[0,2],:] * A[[0,2],:,:], dim=0)  # 

#s1 = w[1]*A[0]
#s2 = w[1]*A[1]
#s3 = w[1]*A[2]
#sm_new = s1+s2+s3


# states_mein = states
# states = states_original
   
#offset = 0
    
#for i in range(rnn_steps):

#  masked_states = states
  
#  ch = masked_states.view(-1, nhid).mm(_Ws[i]).view(i+1, -1, 2*nhid) 
#  c, h = torch.split(ch, nhid, dim=-1)
   
#  c = c.sigmoid()

#  s = torch.zeros_like(s0) 
  #s = s.unsqueeze(0)

#  for k, name in enumerate(PRIMITIVES_rnn): # geht von 0:4
      # switch_rnn = switches_rnn[0]
      # k=4
      #print('ctn')
      #print(count_false)
      #if rnn_switch_row[k]:
        
    #name = PRIMITIVES_rnn[k]
        
#    if name == 'none':
#        continue
#    fn = _get_activation(name)
    # fn = _get_activation(name)

#    unweighted = states + c * (fn(h) - states)
          
#    s += torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)  
    #s += torch.sum(probs[[28,29,39], [0,1,0]].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)  
    #s += sum(probs[row, k].unsqueeze(-1).unsqueeze(-1) * unweighted for row, k in zip(rows[i], cols[i]))

    # else:
     #  count_false += 1
    
#  states = torch.cat([states, s.unsqueeze(0)], 0) 
#  offset += i+1
