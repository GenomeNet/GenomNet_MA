#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 21:50:18 2021

@author: amadeu
"""

### This file is identical to model_searcy.py, EXCEPT that it loads model_de instead of model_discCNN
### This is specifically for DEP-DARTS

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts_tools.model_de import DARTSCell, RNNModel



from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype
from darts_tools.comp_aux import compute_positions, activ_fun

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx, switch_rnn):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    
    self.bn = nn.BatchNorm1d(nhid, affine=False)
    self.switch_rnn = switch_rnn
    
    self.rows, self.nodes, self.cols = compute_positions(self.switch_rnn, rnn_steps, PRIMITIVES_rnn)

    self.acn, self.acf = activ_fun(rnn_steps, PRIMITIVES_rnn, self.switch_rnn)  

  def cell(self, x, h_prev, x_mask, h_mask):
      

    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0)
    probs = F.softmax(self.weights, dim=-1).unsqueeze(-1).unsqueeze(-1)
    
    states = s0.unsqueeze(0) 
    
    offset=0
    
    for i in range(rnn_steps):
        if self.training: 
            masked_states = states * h_mask.unsqueeze(0) 
        else:
            masked_states = states
        
        ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid) 
      
        c, h = torch.split(ch, self.nhid, dim=-1)
        
        c = c.sigmoid()
        
        s = torch.zeros_like(s0) # [2,256]
        for k, name in enumerate(self.acn[i]):
          
            fn = self._get_activation(name)
         
            unweighted = states + c * (fn(h) - states) # states [3,2,256], where s always [2,256]
              
            s += torch.sum(probs[self.rows[offset], self.cols[offset]] * unweighted[self.nodes[offset], :, :], dim=0) 
            offset+=1
           
        s = self.bn(s) 
        states = torch.cat([states, s.unsqueeze(0)], 0) 
    output = torch.mean(states[-CONCAT:], dim=0) 
    return output
    
    
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
  
    def _initialize_arch_parameters(self):

      # alphas for cnn
      k_cnn = sum(1 for i in range(self._steps) for n in range(2+i))
       
      num_ops_cnn = self.num_ops_cnn
        
      self.alphas_normal = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_cnn, num_ops_cnn)))
      self.alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_cnn, num_ops_cnn)))  
        
      # alphas for rnn
      k_rnn = sum(i for i in range(1, rnn_steps+1)) # 1+2+3+4+5+6+7+8=36
        
      num_ops_rnn = self.num_ops_rnn
    
         
      self.weights = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_rnn, num_ops_rnn)))
    
      self.alphas_rnn = self.weights
      for rnn in self.rnns:
          rnn.weights = self.weights
          
      self._arch_parameters = [
          self.alphas_normal,
          self.alphas_reduce,
          self.alphas_rnn,
       ]
            
    def arch_parameters(self):
        return self._arch_parameters


    def genotype(self):

      
      def _parse_cnn(weights):
            gene_cnn = []
            n = 2
            start = 0
            for i in range(self._steps):
                # i=1
                
                end = start + n 
                W = weights[start:end].copy() # first two lines ofalphas
              
                ## here it is determined which connectiosn / inputs of the 2 possibilities at i=0 / resp. 3 possibilities at i=1 / resp. 4 possibilities at i=2
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES_cnn.index('none')))[:2] # just returns edges; at i=0 we have [0,1]
                
                ## now determine which operation of the 8 possibilities
                for j in edges: # for each of the connections, so j=0 and j=1 for i=0
                  k_best = None
                  # find the highest valued operation (highest of 8 columns) for given row j
                  for k in range(len(W[j])): # für jede der 8 Spalten also k=0,1,2,3,4,5,6,7

                    if k != PRIMITIVES_cnn.index('none'): 
                      if k_best is None or W[j][k] > W[j][k_best]: # same line j=1, and see if
                        k_best = k 
                          # Row_alpha=j=0 and column_alpha=k=0: k_best=k=0
                          # j=0 and k=1: compare in row 8 0 of alpha, the column 1 with column 2 and that one becomes k_best
                           
                  gene_cnn.append((PRIMITIVES_cnn[k_best], j))
                start = end
                n += 1
            return gene_cnn
            
      gene_normal = _parse_cnn(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
            
      gene_reduce = _parse_cnn(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
            
      concat = range(2+self._steps-self._multiplier, self._steps+2)
      
      
      def _parse_rnn(probs):
        gene_rnn = []
        start = 0
        for i in range(rnn_steps):
          # i=3 ranges from start0 to end1; i=1 ranges from start1 to end3
          end = start + i + 1 # for i=0 start=0 end=1 
          W = probs[start:end].copy() # only first line with i=0; row 2 and row 3 at i=1
          
          ## determine best connection/Input/Edge: at i=0 that can be j=0 sein, so 0th Row of W is inspected, which is also
          ## 0th row of probs; i=1 is then j=0 again, so 0th row of W is inspected which, however, is 1st row of probs!!!
          j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES_rnn.index('none')))[0]
          
          k_best = None
          ## determine best column/operation
          for k in range(len(W[j])): # iterate over k=0,1,2,3,4
            if k != PRIMITIVES_rnn.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene_rnn.append((PRIMITIVES_rnn[k_best], j))
          start = end
        return gene_rnn

      gene_rnn = _parse_rnn(F.softmax(self.weights, dim=-1).data.cpu().numpy())
    
     
      genotype = Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat,
                                rnn=gene_rnn, rnn_concat=range(rnn_steps+1)[-CONCAT:])
      
      
      return genotype
