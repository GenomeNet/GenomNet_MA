#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 20:47:00 2021

@author: amadeu
"""

from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, CONCAT, Genotype
import numpy as np

import copy

import torch
import torch.nn as nn


def mask2geno(mask):
    
    cnn_normal = mask[0]
    cnn_reduce = mask[1]
    rhn_mask = mask[2]
      
    def _parse_cnn(cnn_mask):
        gene_cnn = []
        n = 2
        start = 0
        for i in range(4):
            # i=0
                
            end = start + n 
            m = cnn_mask[start:end].copy() # die ersten beiden Zeilen von alphas
              
            ## hier bestimmt er welche verbindung/Input von den 2 möglichen bei i=0, bzw. 3 möglichen bei i=1, bzw. 4 möglichen bei i=2
            edges, ops = np.nonzero(m)
            
            for j,h in zip(ops,edges): #
                gene_cnn.append((PRIMITIVES_cnn[j], h))
            
            start = end
            n += 1
        return gene_cnn
    
            
    def _parse_rhn(rhn_mask):
        gene_rhn = []
        n = 1
        start = 0
        for i in range(8):
            # i=0
            end = start + n 
            m = rhn_mask[start:end].copy() # die ersten beiden Zeilen von alphas
              
            ## hier bestimmt er welche verbindung/Input von den 2 möglichen bei i=0, bzw. 3 möglichen bei i=1, bzw. 4 möglichen bei i=2
            edges, ops = np.nonzero(m)
            
            for j, h in zip(ops, edges): #
                gene_rhn.append((PRIMITIVES_rnn[j], h))
            
            start = end
            n += 1
        return gene_rhn
            
    gene_normal = _parse_cnn(cnn_normal)            
    gene_reduce = _parse_cnn(cnn_reduce)
    gene_rnn = _parse_rhn(rhn_mask)
        
    concat = range(2+4-4, 4+2)
     
    genotype = Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat, rnn=gene_rnn, rnn_concat=range(8+1)[-CONCAT:])
      
    return genotype





#####################################
###### used by OSP-NAS

def mask2switch(normal_mask, reduce_mask, rnn_mask):
    
    #normal_mask = supernet_mask[0]
    switches_normal = []
    for i in range(14):
        switch_row = []
        for j in range(len(PRIMITIVES_cnn)):
            if normal_mask[i][j] != 0:
                switch_row.append(True)
            else:
                switch_row.append(False)
                
        switches_normal.append(switch_row)
    switches_normal = copy.deepcopy(switches_normal)
        
    #reduce_mask = supernet_mask[1]
    switches_reduce = []
    for i in range(14):
        switch_row = []
        for j in range(len(PRIMITIVES_cnn)):
            if reduce_mask[i][j] != 0:
                switch_row.append(True)
            else:
                switch_row.append(False)
                
        switches_reduce.append(switch_row)
    switches_reduce = copy.deepcopy(switches_reduce)

    #rnn_mask = supernet_mask[2]
    switches_rnn = []
    for i in range(36):
        switch_row = []
        for j in range(len(PRIMITIVES_rnn)):
            if rnn_mask[i][j] != 0:
                switch_row.append(True)
            else:
                switch_row.append(False)
                
        switches_rnn.append(switch_row)
    switches_rnn = copy.deepcopy(switches_rnn)

        
    return switches_normal, switches_reduce, switches_rnn
