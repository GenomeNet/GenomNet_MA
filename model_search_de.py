#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 21:50:18 2021

@author: amadeu
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts_tools.model_de import DARTSCell, RNNModel



from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype
from darts_tools.comp_aux import compute_positions, activ_fun

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




#import cnn_eval

# for search
class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx, switch_rnn):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    
    self.bn = nn.BatchNorm1d(nhid, affine=False)
    self.switch_rnn = switch_rnn
    
    # probs = model.arch_parameters()[2]
    # hidden, x = torch.rand(2,256), torch.rand(2,256)
    
    self.rows, self.nodes, self.cols = compute_positions(self.switch_rnn, rnn_steps, PRIMITIVES_rnn)
    # switch_rnn = switches_rnn
    # rows, nodes, cols = compute_positions(switch_rnn, rnn_steps, PRIMITIVES_rnn)

    self.acn, self.acf = activ_fun(rnn_steps, PRIMITIVES_rnn, self.switch_rnn)  
    # acn, acf = activ_fun(rnn_steps, PRIMITIVES_rnn, switch_rnn)  

  def cell(self, x, h_prev, x_mask, h_mask):
      

    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0)
    probs = F.softmax(self.weights, dim=-1).unsqueeze(-1).unsqueeze(-1)
    
    # probs = F.softmax(alphas_rnn, dim=-1) 
    
    states = s0.unsqueeze(0) 
    
    offset=0
    
    for i in range(rnn_steps):
        # i=7
        if self.training: 
            masked_states = states * h_mask.unsqueeze(0) 
        else:
            masked_states = states
        
        ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid) 
      
        c, h = torch.split(ch, self.nhid, dim=-1)
        
        c = c.sigmoid()
        
        s = torch.zeros_like(s0) # [2,256]
        #s = s.unsqueeze(0)
        #unweighteds = [states + c * (torch.tanh(h) - states), states + c * (torch.sigmoid(h) - states), states + c * (torch.relu(h) - states), states + c * (h - states)]
        for k, name in enumerate(self.acn[i]):
          
            fn = self._get_activation(name)
         
            unweighted = states + c * (fn(h) - states) # states [3,2,256], wobei s immer [2,256]
              
            s += torch.sum(probs[self.rows[offset], self.cols[offset]] * unweighted[self.nodes[offset], :, :], dim=0) 
            offset+=1
           
        s = self.bn(s) 
        states = torch.cat([states, s.unsqueeze(0)], 0) 
    # offset += i+1
    output = torch.mean(states[-CONCAT:], dim=0) 
    return output
    
    
    return output


class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args)
        
        self._args = args
        # self._initialize_arch_parameters()
        
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
                W = weights[start:end].copy() # die ersten beiden Zeilen von alphas
              
                ## hier bestimmt er welche verbindung/Input von den 2 möglichen bei i=0, bzw. 3 möglichen bei i=1, bzw. 4 möglichen bei i=2
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES_cnn.index('none')))[:2] # gibt einfach nur edges zurück; bei i=0 haben wir [0,1]
                
                ## jetzt bestimmt er, welche operation von den 8 möglichen
                for j in edges: # für jede der Verbindungen also j=0 und j=1 bei i=0
                  # j=0
                  k_best = None
                  # er findet hier die maximales operation (höchste der 8 spalten) für gegebene Zeile j
                  for k in range(len(W[j])): # für jede der 8 Spalten also k=0,1,2,3,4,5,6,7
                    # k=3
              
                    if k != PRIMITIVES_cnn.index('none'): 
                      if k_best is None or W[j][k] > W[j][k_best]: # sind in gleicher Zeile j=1, und schauen ob
                        k_best = k 
                        ## j=0
                          # Zeile_alpha=j=0 und spalte_alpha=k=0: k_best=k=0
                          # j=0 und k=1: vergleich in Zeile 0 von alpha, die Spalte 1 mit spalte 2 und dieser wird k_best
                           
                  gene_cnn.append((PRIMITIVES_cnn[k_best], j))
                start = end
                n += 1
            return gene_cnn
            
      gene_normal = _parse_cnn(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
      #gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
            
      gene_reduce = _parse_cnn(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
      #gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())
            
      concat = range(2+self._steps-self._multiplier, self._steps+2)
      
      
      def _parse_rnn(probs):
        gene_rnn = []
        start = 0
        for i in range(rnn_steps):
          # i=3 geht von start0 bis end1; i=1 geht von start1 bis end3
          end = start + i + 1 # für i=0 start=0 end=1 
          W = probs[start:end].copy() # nur erste zeile bei i=0; zeile 2 und zeile 3 bei i=1
          
          ## bestimme beste Verbindung/Inputmöglichkeit/Edge: bei i=0 kann j=0 sein, also 0te Zeile von W wird angeschaut was auch
          ## 0te Zeile von probs ist; i=1 ist dann wieder j=0, also 0te Zeile vo nW wird angeschaut was aber 1te Zeile von probs ist!!!
          j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES_rnn.index('none')))[0]
          
          k_best = None
          ## bestimme beste spalte/operation
          for k in range(len(W[j])): # iteriere über k=0,1,2,3,4
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
