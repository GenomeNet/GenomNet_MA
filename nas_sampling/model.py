#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:28:13 2021

@author: amadeu
"""

from generalNAS_tools.operations_14_9 import OPS, ReLUConvBN, FactorizedReduce
from generalNAS_tools.genotypes import PRIMITIVES_cnn, rnn_steps
from generalNAS_tools.training_utils import mask2d

import numpy as np
from torch.autograd import Variable

import torch
import torch.nn as nn

INITRANGE = 0.04

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MixedOp(nn.Module):
    '''mask: op'''
    
    def __init__(self, C, stride, mask_cnn):
        super(MixedOp, self).__init__()
        self.stride = stride
        self._ops = nn.ModuleList()
        
        mask_1 = np.nonzero(mask_cnn)[0] # 
      
        self._super_mask = mask_1 
                    
        for i in range(len(mask_cnn)):
            if mask_cnn[i] != 0:
                primitive = PRIMITIVES_cnn[i] 
                op = OPS[primitive](C, stride, False)

                if 'pool' in primitive: #
                    op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
               
                
                self._ops.add_module(str(i), op)
            else:
                op = None

                self._ops.add_module(str(i), op)
                    
    
    def forward(self, x, mask_cnn):
        if (mask_cnn==0).all(): 
            # if normal cell
            if self.stride == 1:
                return torch.zeros_like(x)
            return torch.zeros_like(x[:,:,::self.stride]) 
        else: 
            result = 0
            mask_2 = np.nonzero(mask_cnn)[0] 
            if len(mask_2) != 0:
                for selected in np.nditer(mask_2): 
                    result += self._ops[selected](x)
                    
            return result


class CNN_Cell_search(nn.Module):
    '''mask: 14 * 8'''

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, mask_cnn, reduction_high):
        super(CNN_Cell_search, self).__init__()
        
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        cnt = 0
        
        for i in range(self._steps):
            for j in range(2 + i):
                
                stride = 2 if reduction and j < 2 else 1
                    
                if (reduction==False) or j >= 2:
                    stride=1
                
                if reduction and j < 2:
                    stride=2
                    
                if reduction_high and j < 2:
                    stride=3
                
                op = MixedOp(C, stride, mask_cnn[cnt])
                
                self.cell_ops.add_module(str(cnt), op)
                
                cnt += 1
                
    def forward(self, s0, s1, mask_cnn):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.cell_ops[offset + j].forward(h, mask_cnn[offset + j]) for j, h in enumerate(states))
        
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)



# for evaluation
class DARTSCell(nn.Module):

  def __init__(self, ninp, nhid, dropouth, dropoutx, genotype):
    super(DARTSCell, self).__init__()
    self.nhid = nhid
    self.dropouth = dropouth
    self.dropoutx = dropoutx
    self.genotype = genotype

    steps = len(self.genotype.rnn) if self.genotype is not None else rnn_steps
    self._W0 = nn.Parameter(torch.Tensor(ninp+nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) 
    self._Ws = nn.ParameterList([
        nn.Parameter(torch.Tensor(nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) for i in range(steps) 
    ])
   
  def forward(self, inputs, hidden, rnn_mask):
  
    T, B = inputs.size(0), inputs.size(1) # 10,2

    if self.training:
        
      x_mask = mask2d(B, inputs.size(2), keep_prob=1.-self.dropoutx) 

      h_mask = mask2d(B, hidden.size(2), keep_prob=1.-self.dropouth)

    else:
        
      x_mask = h_mask = None
      
    hidden = hidden[0]

    hiddens = []
    for t in range(T):
 
      hidden = self.cell(inputs[t], hidden, x_mask, h_mask, rnn_mask)

      hiddens.append(hidden)
    hiddens = torch.stack(hiddens)
    return hiddens, hiddens[-1].unsqueeze(0)

  def _compute_init_state(self, x, h_prev, x_mask, h_mask):

    if self.training:
      xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1) # entlang der channels zusammenfügen
    else:
      xh_prev = torch.cat([x, h_prev], dim=-1)
      
    c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1) 


    c0 = c0.sigmoid() 
    h0 = h0.tanh() 
    s0 = h_prev + c0 * (h0-h_prev)
    return s0

  def _get_activation(self, name):
    if name == 'tanh':
      f = torch.tanh
    elif name == 'relu':
      f = torch.relu
    elif name == 'sigmoid':
      f = torch.sigmoid
    elif name == 'identity':
      f = lambda x: x
    else:
      raise NotImplementedError
    return f

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    states = [s0]
    for i, (name, pred) in enumerate(self.genotype.rnn):
      s_prev = states[pred]
      if self.training:
        ch = (s_prev * h_mask).mm(self._Ws[i])
      else:
        ch = s_prev.mm(self._Ws[i])
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()
      fn = self._get_activation(name)
      h = fn(h)
      s = s_prev + c * (h-s_prev)
      states += [s]
    output = torch.mean(torch.stack([states[i] for i in self.genotype.concat], -1), -1)
    return output

class RNNModel(nn.Module):

   
    def __init__(self, seq_len, 
                 dropouth=0.5, dropoutx=0.5,
                 C=8, num_classes=4, layers=3, steps=4, multiplier=4, stem_multiplier=3,
                 search=True, drop_path_prob=0.2, genotype=None, task=None, mask=None):
                
        super(RNNModel, self).__init__()
    
        assert search  # search == FALSE is not implemented here!
        
        self.mask_normal = mask[0]
        self.mask_reduce = mask[1]

        self.mask_rnn = mask[2]
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

        self.dropouth = dropouth
        self.dropoutx = dropoutx
        
    
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv1d(4, C_curr, 13, padding=6, bias=False),
            nn.BatchNorm1d(C_curr)
        )
    
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        
        layer_list = []
        
        for i in range(layers):
            layer_list.append(i)
        
        normal_cells = layer_list[::3]
        
        num_neurons = seq_len # =1000
        
        for i in range(layers):
       
            if i not in normal_cells:
                
                C_curr *= 2
                reduction = True
                
                if (i==5):
                    reduction_high=True
                    num_neurons = round(num_neurons/3)
                    #stride=3
                else:
                    reduction_high=False
                    num_neurons = round(num_neurons/2)
                    
                cell = CNN_Cell_search(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.mask_reduce, reduction_high)

            else:
                reduction = reduction_high = False
                cell = CNN_Cell_search(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.mask_normal, reduction_high) 
              
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.drop_path_prob = drop_path_prob
        
        out_channels = C_curr*steps
        
        ninp, nhid, nhidlast = out_channels, out_channels, out_channels
    
    
            
        assert ninp == nhid == nhidlast
        # again, we have different rnn cells for search and for evaluation
        # run search
        assert genotype is None
        from nas_sampling import model_search
        cell_cls = model_search.DARTSCellSearch
        self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx)]
        
        
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.decoder = nn.Linear(num_neurons*out_channels, 925) # because we have flattened 
        
        self.classifier = nn.Linear(925, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.dropout_lin = nn.Dropout(p=0.5)

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        
        self.cell_cls = cell_cls
        
        if task == "TF_bindings":
            
            self.final = nn.Sigmoid()
            
        else:
            
            self.final = nn.Identity()

        
    def init_weights(self):
        
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)  

    def forward(self, input, hidden, mask, return_h=False):
        
        s0 = s1 = self.stem(input)
        
        for i, cell in enumerate(self.cells):   
            
            if cell.reduction:
                cur_mask = mask[1]
            else:
                cur_mask = mask[0]
            s0, s1 = s1, cell(s0, s1, cur_mask)

       
        batch_size = s1.size(0) 

        out_channels = s1.size(1)
        num_neurons = s1.size(2)
        
        # CNN expects [batchsize, input_channels, signal_length]
        # RHN expect [seq_len, batch_size, features]
        
        x = s1.permute(2,0,1)

        raw_output = x
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns): 
            
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l], mask[2])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            
        hidden = new_hidden

        output = raw_output
        
        outputs.append(output)
        
        output = output.permute(1,0,2)

        
        # flatten the RNN output
        x = torch.flatten(output, start_dim= 1) 
        
        # dropout layer
        x = self.dropout_lin(x)
        
        # linear layer
        x = self.decoder(x) 
        
        # dropout layer
        #x = self.dropout_lin(x)
        
        x = self.relu(x)
        
        # linear layer
        x = self.classifier(x)

        logit = self.final(x)

        if return_h:
            return logit, hidden, raw_outputs, outputs
        
        return logit, hidden 
     

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()
    
    def _loss(self, hidden, input, target):
        log_prob, hidden_next = self(input, hidden, return_h=False) 
          
        criterion = nn.CrossEntropyLoss()
        loss = criterion(log_prob, target)
        
        return loss, hidden_next
  
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [Variable(weight.new(1, bsz, self.nhid).zero_())]
