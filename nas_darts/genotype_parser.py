#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:50:59 2021

@author: amadeu
"""



from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype
import copy
import torch
import torch.nn.functional as F
import numpy as np




_steps = _multiplier = 4
concat = range(2+_steps-_multiplier, _steps+2)


#row = F.softmax(arch_rnn[25])
#row2 = F.softmax(new_arch_rnn[22])

#row1 = F.softmax(arch_rnn[21])
#row12 = F.softmax(new_arch_rnn[20])

def parse_genotype(switches_normal_cnn, arch_normal, switches_reduce_cnn, arch_reduce, switches_rnn, arch_rnn):
    
        alphas_rnn = F.softmax(arch_rnn, dim=-1)#.data.cpu().numpy()
        # alphas_rnn = F.softmax(alphas_rnn, dim=-1).data.cpu().numpy()

        # alphas_rnn = alphas_rnn.data.cpu().numpy()
        # alphas_rnn = np.copy(arch_rnn.data.cpu().numpy())
            
        switches_rnn = copy.deepcopy(switches_rnn)
        rnn_final = [0 for idx in range(36)]
        switch_count_rnn = 0
        for i in range(len(switches_rnn)): 
             # i=30
             if switches_rnn[i].count(False) != len(switches_rnn[i]): #wenn nicht alle false sind -> keine discarded edge
                 if switches_rnn[i][0] == True:
                     # d.h. wir bekommen die 14 max values raus von jeder edge: jetzt würde ich einfach 
                     # erstmal von dem letzten Node (welche 5 edges hat) die schlechteste discarden, indem wir die gesamte switches 
                     # Zeile False setzten
                     alphas_rnn[i-switch_count_rnn][0] = 0
                 rnn_final[i] = max(alphas_rnn[i-switch_count_rnn]) # add best/max operation to final_prob 
             else: # if previous discarded edge
                 rnn_final[i] = 0 # set previous discarded edge to 1 (so that it does not get discarded in this stage)
                 switch_count_rnn += 1
            
            
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
            
        def _parse_rnn(switches_rnn, alphas_rnn, rnn_final):
             keep_edge = []    
             start=0
             gene_rnn = []
             disc_edge_cnt = 0
             n = 1
             # for i in range(len(switches_rnn)):
             for i in range(8):
                 # i=0
                 end = start + i + 1
                 tbrnn = rnn_final[start:end]
                 if i>0:
                     edge_rnn = sorted(range(n), key=lambda x: tbrnn[x]) 
                     keep_edge.append(edge_rnn[-1]+start)                     
                 else:
                     keep_edge.append(i)
                 
                 for j in range(start,end):
                     # j=0
                     if j in keep_edge:
                         alpha_row = alphas_rnn[j-disc_edge_cnt] # torch.Tensor(alphas_rnn[j-disc_edge_cnt])
                         op_idxs = alpha_row.sort()[1]
                         idxs = []
                         for r in range(len(PRIMITIVES_rnn)):
                             if switches_rnn[j][r]:
                                 idxs.append(r)
                         
                         k_best = op_idxs[-1]
                         op_idx = idxs[k_best]
                         
                         if op_idx == 0:
                             k_best = op_idxs[-2]
                             op_idx = idxs[k_best]
                         else: 
                             op_idx = op_idx
                                                     
                         gene_rnn.append((PRIMITIVES_rnn[op_idx], j-start))
                         
                     if switches_rnn[j].count(False) == len(switches_rnn[j]): # discarded edge
                         disc_edge_cnt += 1
                         
                 
                 start = end
                 n +=1
             
             return gene_rnn
         
        gene_rhn = _parse_rnn(switches_rnn, alphas_rnn, rnn_final)

        alphas_normal = F.softmax(arch_normal, dim=-1)#.data.cpu().numpy()
        # alphas_normal = alphas_normal.data.cpu().numpy()
        # alphas_normal = np.copy(arch_normal.data.cpu().numpy())

        switches_normal_cnn = copy.deepcopy(switches_normal_cnn)
        normal_final = [0 for idx in range(14)]
        switch_count_cnn = 0
        for i in range(len(switches_normal_cnn)): 
             # i=4
             if switches_normal_cnn[i].count(False) != len(switches_normal_cnn[i]): #wenn nicht alle false sind -> keine discarded edge
                 if switches_normal_cnn[i][0] == True:
                     alphas_normal[i-switch_count_cnn][0] = 0
                 # d.h. wir bekommen die 14 max values raus von jeder edge: jetzt würde ich einfach 
                 # erstmal von dem letzten Node (welche 5 edges hat) die schlechteste discarden, indem wir die gesamte switches 
                 # Zeile False setzten
                 normal_final[i] = max(alphas_normal[i-switch_count_cnn]) # add best/max operation to final_prob 
             else: # if previous discarded edge
                 normal_final[i] = 0 # set previous discarded edge to 1 (so that it does not get discarded in this stage)
                 switch_count_cnn += 1
          
        alphas_reduce = F.softmax(arch_reduce, dim=-1)#.data.cpu().numpy()
        # alphas_reduce = np.copy(arch_reduce.data.cpu().numpy())
        # alphas_reduce = alphas_reduce.data.cpu().numpy()

        switches_reduce_cnn = copy.deepcopy(switches_reduce_cnn)
        reduce_final = [0 for idx in range(14)]
        switch_count_cnn = 0
        for i in range(len(switches_reduce_cnn)): 
             # i=30
             if switches_reduce_cnn[i].count(False) != len(switches_reduce_cnn[i]): #wenn nicht alle false sind -> keine discarded edge
                 if switches_reduce_cnn[i][0] == True:
                     alphas_reduce[i-switch_count_cnn][0] = 0
                 # d.h. wir bekommen die 14 max values raus von jeder edge: jetzt würde ich einfach 
                 # erstmal von dem letzten Node (welche 5 edges hat) die schlechteste discarden, indem wir die gesamte switches 
                 # Zeile False setzten
                 reduce_final[i] = max(alphas_reduce[i-switch_count_cnn]) # add best/max operation to final_prob 
             else: # if previous discarded edge
                 reduce_final[i] = 0 # set previous discarded edge to 1 (so that it does not get discarded in this stage)
                 switch_count_cnn += 1
                             
        
        def _parse_cnn(switches_cnn, alphas_cnn, cnn_final):
             keep_edge = []    
             start=0
             gene_cnn = []
             disc_edge_cnt = 0
             n = 2
             # for i in range(len(switches_rnn)):
             for i in range(4):
                 # i=0
                 end = start + n
                 tbcnn = cnn_final[start:end]
                 if i>0:
                     edge_cnn = sorted(range(n), key=lambda x: tbcnn[x]) 
                     keep_edge.append(edge_cnn[-1] + start) 
                     keep_edge.append(edge_cnn[-2] + start)
                 else:
                     keep_edge.append(start)
                     keep_edge.append(end - 1)
                 
                 for j in range(start,end):
                     # j=0
                     if j in keep_edge:
                         alpha_row = alphas_cnn[j-disc_edge_cnt]
                         op_idxs = alpha_row.sort()[1]
                         idxs = []
                         for r in range(len(PRIMITIVES_cnn)):
                             if switches_cnn[j][r]:
                                 idxs.append(r)
                         
                         k_best = op_idxs[-1]
                         op_idx = idxs[k_best]
                         
                         if op_idx == 0:
                             k_best = op_idxs[-2]
                             op_idx = idxs[k_best]
                         else: 
                             op_idx = op_idx
                                                     
                         gene_cnn.append((PRIMITIVES_cnn[op_idx], j-start))
                         
                     if switches_cnn[j].count(False) == len(switches_cnn[j]): # discarded edge
                         disc_edge_cnt += 1  
                 
                 start = end
                 n +=1
             
             return gene_cnn
         
        gene_normal = _parse_cnn(switches_normal_cnn, alphas_normal, normal_final)
        gene_reduce = _parse_cnn(switches_reduce_cnn, alphas_reduce, reduce_final)
        
        genotype = Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat,
                                rnn=gene_rhn, rnn_concat=range(rnn_steps+1)[-CONCAT:])
        
        
        return genotype