#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 09:58:01 2021

@author: amadeu
"""

import random
import numpy as np
import copy
#####################################
###### used by OSP-NAS


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
    n = 2
    start = 0
    edge_sampler = np.empty((0,1), int)
    for i in range(4):  
        end = start + n
        masks = []
        cnt = 0
        for j in range(start, end):
            mask = np.nonzero(cnn_gene[j])[0]#[0] # "blocked elements"
            if mask.size == 0:
                cnt += 1
            masks.append(mask)
        num_edges = len(masks)
        active_edges = num_edges - cnt
        
        # if more than 3 edges of a node are active, it is allowed to add all edges of this node the edge sampler
        if active_edges >= 3:
            # add all edges except those which are empty
            for e in range(num_edges):
                if masks[e].size != 0:
                    edge_sampler = np.append(edge_sampler, np.array(e+start)) 

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
        edge_sampler = create_cnn_edge_sampler(cnn_gene) # gives us the edges that we sample from
        if edge_sampler.size != 0:

            # random sampling of an edge
            random_cnn_edge = random.choice(edge_sampler)
            
            # random sampling of an operation 
            ops_idxs = np.nonzero(cnn_gene[random_cnn_edge])[0] # gives us the operations that we can sample from
            random_cnn_op = random.choice(ops_idxs)
            
            # discard corresponding operation to build new supersubnet
            cnn_gene[random_cnn_edge, random_cnn_op] = 0 
            disc_ops.append([random_cnn_edge,random_cnn_op])
            
    return cnn_gene, disc_ops

  
# RHN
def create_rhn_edge_sampler(rhn_gene):
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
            mask = np.nonzero(rhn_gene[j])[0] # second 0 only works when we have 1 element remaining
            if mask.size == 0:
                cnt += 1
            masks.append(mask)
        
        num_edges = len(masks)
        activate_edges = num_edges - cnt
        
        if activate_edges >=2:
            
            for e in range(num_edges):
                if masks[e].size != 0:
                    edge_sampler = np.append(edge_sampler, np.array(e+start))
        else:
            
            for e in range(num_edges):
                num_ops = len(masks[e])
                if num_ops > 1:
                    edge_sampler = np.append(edge_sampler, np.array(e+start))
    
        start = end
        n = n + 1
    return edge_sampler


# used by OSP-NAS
def create_rhn_supersubnet(supernet_mask, num_disc):
    rhn_gene = copy.deepcopy(supernet_mask[2])
    disc_ops = []
    for i in range(num_disc):
        edge_sampler = create_rhn_edge_sampler(rhn_gene)
        if edge_sampler.size != 0:

            random_rhn_edge = random.choice(edge_sampler)
            # random sampling of an operation 
            ops_idxs = np.nonzero(rhn_gene[random_rhn_edge])[0] # ist empty, when all are 0
            random_rhn_op = random.choice(ops_idxs)
            
            rhn_gene[random_rhn_edge, random_rhn_op] = 0
            disc_ops.append([random_rhn_edge, random_rhn_op])
            
    return rhn_gene, disc_ops


def create_final_cnn_edge_sampler(cnn_gene):
    n = 2
    start = 0
    edge_sampler = np.empty((0,1), int)
    for i in range(4):  
        end = start + n
        masks = []
        cnt = 0
        for j in range(start, end):
            mask = np.nonzero(cnn_gene[j])[0] # "blocked elements"
            if mask.size == 0:
                cnt += 1
            masks.append(mask)
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
                num_ops = len(masks[e])
                if num_ops > 1:
                    edge_sampler = np.append(edge_sampler, np.array(e+start))
    
        start = end
        n = n + 1
        
    return edge_sampler 
        


