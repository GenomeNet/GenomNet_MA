#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 09:58:01 2021

@author: amadeu
"""


import random

import numpy as np
import time

from generalNAS_tools.genotypes import OPS_cnn, OPS_rnn, PRIMITIVES_cnn, PRIMITIVES_rnn, Genotype
#from random_sample import generate_random_architectures
#from utils import merge
import copy




def maskout_ops(disc_ops_normal, disc_ops_reduce, disc_ops_rhn, supernet_mask):
    supernet_mask = copy.deepcopy(supernet_mask)
    supernet_mask_new = supernet_mask
    supernet_mask_new[0][disc_ops_normal[0], disc_ops_normal[1]] = 0
   
    supernet_mask_new[1][disc_ops_reduce[0], disc_ops_reduce[1]] = 0
    
    supernet_mask_new[2][disc_ops_rhn[0], disc_ops_rhn[1]] = 0
   
    return supernet_mask_new


def create_new_supersubnet(hb_results, supernet_mask):
    supernet_mask = copy.deepcopy(supernet_mask)
    supernet_mask_new = supernet_mask

    for i in range(len(hb_results)-1):
        disc_ops = hb_results[i][0]
        
        supernet_mask_new[0][disc_ops[0][0], disc_ops[0][1]] = 0
        
        supernet_mask_new[1][disc_ops[1][0], disc_ops[1][1]] = 0
      
        supernet_mask_new[2][disc_ops[2][0], disc_ops[2][1]] = 0
       
    return supernet_mask_new




# CNN
def create_cnn_edge_sampler(cnn_gene):
    # cnn_gene = supernet_mask[0]
    n = 2
    start = 0
    edge_sampler = np.empty((0,1), int)
    #remain_cnn_ops = []
    for i in range(4):  
        # i = 0
        end = start + n
        masks = []
        cnt = 0
        for j in range(start, end):
            # j =1
            mask = np.nonzero(cnn_gene[j])[0]#[0] # "blocked elements"
            if mask.size == 0:
                cnt += 1
            masks.append(mask)
            #count += 1
        #remain_cnn_ops.append(masks)
        # num_edges = len(remain_cnn_ops[i])
        num_edges = len(masks)
        active_edges = num_edges - cnt
        
        # if more than 3 edges of a node are active, it is allowed to add all edges of this node the edge sampler
        if active_edges >= 3:
            # add all edges except those which are empty
            for e in range(num_edges):
                # e = 2
                if masks[e].size != 0:
                    edge_sampler = np.append(edge_sampler, np.array(e+start)) 

                    # edge_sampler = np.append(edge_sampler, np.arange(start, end, step=1)) 
    
        # if only 2 edges are active in a node, we are only allowed to add edges, which have at least one value/operation remained (because each node has 2 edges in final architecture)
        else:
            for e in range(num_edges):
                # e=3
                num_ops = len(masks[e])
                if num_ops > 1:
                    edge_sampler = np.append(edge_sampler, np.array(e+start))
    
        start = end
        n = n + 1
        
    return edge_sampler 
    

    
def create_cnn_supersubnet(supernet_mask, num_disc):
    cnn_gene = copy.deepcopy(supernet_mask)
    disc_ops = []

    for i in range(num_disc):
        edge_sampler = create_cnn_edge_sampler(cnn_gene) # gives us the edges, where we can sample from
        if edge_sampler.size != 0:

            # random sampling of an edge
            random_cnn_edge = random.choice(edge_sampler)
            
            # random sampling of an operation 
            ops_idxs = np.nonzero(cnn_gene[random_cnn_edge])[0]#[0] # gives us the operations, where we can sample from
            random_cnn_op = random.choice(ops_idxs)
            
            # discard corresponding operation to build new supersubnet
            cnn_gene[random_cnn_edge, random_cnn_op] = 0 
            disc_ops.append([random_cnn_edge,random_cnn_op])
            
    return cnn_gene, disc_ops

  
# RHN
def create_rhn_edge_sampler(rhn_gene):
    # rhn_gene = supernet_mask[1]
    n = 1
    start = 0
    edge_sampler = np.empty((0,1), int)
    
    for i in range(8):
        # i = 0
        end = start + n
        masks = []
        cnt = 0
        for j in range(start, end): 
            # j =
            mask = np.nonzero(rhn_gene[j])[0]#[0] # zweites 0 geht nur, wenn wir nur noch 1 Element haben
            if mask.size == 0:
                cnt += 1
            masks.append(mask)
        
        num_edges = len(masks)
        activate_edges = num_edges - cnt
        
        if activate_edges >=2:
            
            for e in range(num_edges):
                # e=3
                if masks[e].size != 0:
                    edge_sampler = np.append(edge_sampler, np.array(e+start))
        else:
            
            for e in range(num_edges):
                # e=1
                num_ops = len(masks[e])
                if num_ops > 1:
                    edge_sampler = np.append(edge_sampler, np.array(e+start))
    
        start = end
        n = n + 1
    return edge_sampler


# 
def create_rhn_supersubnet(supernet_mask, num_disc):
    rhn_gene = copy.deepcopy(supernet_mask[2])
    disc_ops = []
    for i in range(num_disc):
        edge_sampler = create_rhn_edge_sampler(rhn_gene)
        if edge_sampler.size != 0:

            random_rhn_edge = random.choice(edge_sampler)
            # random sampling of an operation 
            ops_idxs = np.nonzero(rhn_gene[random_rhn_edge])[0]#[0] # ist empty, wenn alle 0 sind
            random_rhn_op = random.choice(ops_idxs)
            
            rhn_gene[random_rhn_edge, random_rhn_op] = 0
            disc_ops.append([random_rhn_edge, random_rhn_op])
            
    return rhn_gene, disc_ops

    

#cnt=0
#for i in range(len(cnn_gene)):
    # i=0
#    for j in range(len(cnn_gene[i])):
#        if cnn_gene[i][j]!=0:
#            cnt += 1
            
#if cnt == 9: # if we reached final stage
           
#    edge_sampler = create_cnn_edge_sampler(cnn_gene) # gives us the edges, where we can sample from
    
#    if len(edge_sampler)==1:

#        ops_idxs = np.nonzero(cnn_gene[int(edge_sampler)])[0]#[0] # gives us the operations, where we can sample from

#        for j, op in enumerate(ops_idxs):
#            disc_ops.append([int(edge_sampler), op])
#    else:
        
#        for j, edge in enumerate(edge_sampler):
#            op = np.nonzero(cnn_gene[int(edge)])[0]#
#            disc_ops.append([edge, int(op)])




       

def create_final_cnn_edge_sampler(cnn_gene):
    # cnn_gene = supernet_mask[0]
    n = 2
    start = 0
    edge_sampler = np.empty((0,1), int)
    #remain_cnn_ops = []
    for i in range(4):  
        # i = 0
        end = start + n
        masks = []
        cnt = 0
        for j in range(start, end):
            # j =1
            mask = np.nonzero(cnn_gene[j])[0]#[0] # "blocked elements"
            if mask.size == 0:
                cnt += 1
            masks.append(mask)
            #count += 1
        #remain_cnn_ops.append(masks)
        # num_edges = len(remain_cnn_ops[i])
        num_edges = len(masks)
        active_edges = num_edges - cnt
        
        # if more than 3 edges of a node are active, it is allowed to add all edges of this node the edge sampler
        if active_edges >= 3:
            # add all edges except those which are empty
            for e in range(num_edges):
                # e = 2
                if masks[e].size != 0:
                    edge_sampler = np.append(edge_sampler, np.array(e+start)) 

                    # edge_sampler = np.append(edge_sampler, np.arange(start, end, step=1)) 
    
        # if only 2 edges are active in a node, we are only allowed to add edges, which have at least one value/operation remained (because each node has 2 edges in final architecture)
        else:
            for e in range(num_edges):
                # e=3
                num_ops = len(masks[e])
                if num_ops > 1:
                    edge_sampler = np.append(edge_sampler, np.array(e+start))
    
        start = end
        n = n + 1
        
    return edge_sampler 
        





#def get_final_archs(cnn_gene)
#    n = 2
#    start = 0
#    #remain_cnn_ops = []
#    archs = []
#    for i in range(4):  
        # i = 0
#        end = start + n
#        ops=[]
#        for j in range(start, end):
            # j =4#
#            op = np.nonzero(cnn_gene[j])[0]#[0] # "blocked elements"
#            if op.size == 1:
#                ops.append([j,int(op)])
#            if op.size==2:
#                for s in op:
                    
#                    ops.append([j,int(s)])
#                    num_archs = 2 
            
                
#        archs.append(ops)
#        if len(ops)==3:
#            num_archs=3
            
            
#        start = end
#        n = n + 1
       
#    if num_archs == 3:
#        idxs = [[0,1],[0,2],[1,2]]
        
#        final_archs = []
#        for i in range(num_archs):     
#            final_arch = []
#            for node in archs:
                
#                if len(node) != 2:
#                    idx = idxs[i]
        
#                    multi_ops = node
#                    final_arch.append([multi_ops[idx[0]],multi_ops[idx[1]]])
                    
#                else:
#                    final_arch.append(node)
                    
#            final_archs.append(final_arch)
            
#    if num_archs == 2:
        
      
#for op in multi_ops:
#    print(op[0])

        

