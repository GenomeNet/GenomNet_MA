#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:24:45 2021

@author: amadeu
"""

from predictors.models import GCN
from predictors.utils.gcn_train_val import train, validate
import torch
import torch.optim as optim
from predictors.dataloader import NasDataset
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# dataset=__train_dataset
# val_dataset=__valset
# ifPretrained=ifPretrain
# maxsize=maxsize
# ifsigmoid=True
# num_epoch=2
# lr=lr 
# selected_loss=loss
 


class NeuralNet(object):

    def __init__(self, dataset, val_dataset=None, ifPretrained=False, maxsize=7):
        """Initialization of NeuralNet object
        Keyword arguments:
        architecture -- a tuple containing the number of nodes in each layer
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        self.__dataset = dataset
        self.__val_dataset = val_dataset
        self.gcn = None
        self.ifPretrained = ifPretrained
        self.maxsize = maxsize

    def train(self, lr, num_epoch, selected_loss, ifsigmoid):
        # NasDataset wird mit Dataset aus torch.loader initialisiert, damit ich diesen generator habe, also er mir die samples 
        # als batches raushaut mit for ... enumerate(train_loader)
        dataset = NasDataset(sample=self.__dataset, maxsize=self.maxsize)
        # dataset = NasDataset(sample=__dataset, maxsize=maxsize)
        # dataset.graph
        val_dataset = NasDataset(sample=self.__val_dataset, maxsize=self.maxsize)
        # val_dataset = NasDataset(sample=__val_dataset, maxsize=maxsize)

        batch_size = 128 # batch_size = 2
        # Creating PT data samplers and loaders:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4, )
        # wird erstmal nur initialisiert; adj und feat wird später eingesetzt
        gcn = GCN(
            nfeat_cnn=len(self.__dataset[0]['operations_cnn'][0]) + 1, # nfeat=num_features: anzahl spaltenelemente von operations + 1=7
            nfeat_rhn=len(self.__dataset[0]['operations_rhn'][0]) + 1, # =8

            ifsigmoid=ifsigmoid
        )
        
        gcn = gcn.to(device)
        gcn = torch.nn.DataParallel(gcn)
        optimizer = optim.Adam(gcn.parameters(),
                               lr=lr,
                               )
        loss = selected_loss
        if self.__val_dataset:
            
            validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                            shuffle=True, num_workers=4, )
            
            for i in range(num_epoch):
                train(model=gcn, optimizer=optimizer, loss=loss, train_loader=train_loader, epoch=i)
                validate(model=gcn, loss=loss, validation_loader=validation_loader)
            self.gcn = gcn
            
        else:
            for i in range(num_epoch):
                train(model=gcn, optimizer=optimizer, loss=loss, train_loader=train_loader, epoch=i)
            self.gcn = gcn

    @property
    def network(self):
        return self.gcn
