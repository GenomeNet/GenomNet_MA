#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:28:39 2021

@author: amadeu
"""

import torch
import torch.nn as nn

from generalNAS_tools.genotypes import rnn_steps, PRIMITIVES_rnn, CONCAT

from nas_sampling.model import DARTSCell, RNNModel

import numpy as np

#####################################
###### used by OSP-NAS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ninp, nhid, dropouth, dropoutx, mask
class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    
    self.bn = nn.BatchNorm1d(nhid, affine=False)
   
    
   
  def cell(self, x, h_prev, x_mask, h_mask, rnn_mask):
      
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0)

    states = s0.unsqueeze(0) 
    
    offset = 0

    for i in range(rnn_steps):
       
        if self.training: 
            masked_states = states * h_mask.unsqueeze(0) 
        else:
            masked_states = states
        
        ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid) 
      
        c, h = torch.split(ch, self.nhid, dim=-1)
        
        c = c.sigmoid()
        
        s = torch.zeros_like(s0) # [2,256]
        for k, name in enumerate(PRIMITIVES_rnn):

            if (rnn_mask[offset:offset+i+1, k]==0).all():
                continue
          
            fn = self._get_activation(name)
            
            idxs = np.nonzero(rnn_mask[offset:offset+i+1, k])[0]
         
            unweighted = states + c * (fn(h) - states) # states [3,2,256], where s always [2,256]
            s += torch.sum(unweighted[idxs, :, :], dim=0)
           
        s = self.bn(s) 
        states = torch.cat([states, s.unsqueeze(0)], 0) 
        offset += i+1

    output = torch.mean(states[-CONCAT:], dim=0) 
    return output





class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args)
        
        self._args = args
        
    def new(self):
        model_new = RNNModelSearch(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

         
    def _loss(self, hidden, input, target, criterion):
      log_prob, hidden_next = self(input, hidden, return_h=False) 
      
      loss = criterion(log_prob, target)
      
      return loss, hidden_next
