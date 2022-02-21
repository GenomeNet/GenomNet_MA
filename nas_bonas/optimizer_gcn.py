#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:23:22 2021

@author: amadeu
"""

import time
import numpy as np
from . import neural_net_gcn as nn
from . import linear_regressor as lm

import scipy.stats as stats
from predictors.utils.gcn_utils import padzero, add_global_node
from opendomain_utils.loss_function import weighted_exp, weighted_linear, weighted_log
from opendomain_utils.transform_genotype import transform_Genotype
import torch
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Optimizer(object):

    def __init__(self, dataset, val_set=None, ifPretrain=False, ifTransformSigmoid=True, ifFindMax=True,
                 lr=0.001, train_epoch=1, lossnum=3, maxsize=7):
        """Initialization of Optimizer object
        Keyword arguments:
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        what is architecture
        """
        # when initializing only process_data() is executed
        self.lossList = [torch.nn.MSELoss(), weighted_log, weighted_linear, weighted_exp]
        self.maxsize = maxsize
        self.__dataset = dataset
        self.__valset = val_set
        self.process_data() # trained_arch_list is turned into A and X and saved
        self.ifPretrain = ifPretrain
        self.ifTransformSigmoid = ifTransformSigmoid
        self.ifFindMax = ifFindMax
        self.lr = lr
        self.num_epoch = train_epoch
        self.loss = self.lossList[lossnum]

    def train(self):
        """ Using the stored dataset and architecture, trains the neural net to
        perform feature extraction, and the linear regressor to perform prediction
        and confidence interval computation.
        """
        start = time.time()
        self.__train_dataset = self.__dataset
        # __train_dataset = __dataset

        # just being initialized, no train
        neural_net = nn.NeuralNet(dataset=self.__train_dataset, val_dataset=self.__valset, ifPretrained=self.ifPretrain,
                                  maxsize=self.maxsize) # above neural_net_gcn was imported as nn

        # run train(), where GCN is first initialized and trained for num_epoch. The training
        # works as always, only now with adj and feat as input
        neural_net.train(num_epoch=self.num_epoch, lr=self.lr, selected_loss=self.loss,
                         ifsigmoid=self.ifTransformSigmoid)
        
        self.gcn = neural_net.gcn
        # do embedding with GCN -> i.e. not final outpout of GCN, but embedding output
        # we now have have [4, 128] i.e. concated embedding, instead of  [4,64]
        # which is in fact what I need
        train_features = self.extract_features(self.train_adj_cnn, self.train_features_cnn, self.train_adj_rhn, self.train_features_rhn) 
        lm_dataset = (train_features, self.train_Y)
        # at first initialize with data, doesn't do anything yet
        linear_regressor = lm.LinearRegressor(lm_dataset, intercept=False, ifTransformSigmoid=self.ifTransformSigmoid)
        
        # now train the initialized linear_regressor model
        linear_regressor.train()
        time_ = time.time()
        print(f"train gcn time:{start - time_}")

    def process_data(self):
        print("processing training data")
        self.__train_dataset = self.__dataset
        self.train_adj_cnn = np.array(
            [add_global_node(padzero(np.array(sample['adjacency_matrix_cnn']), True, maxsize=self.maxsize), True) for sample
             in self.__train_dataset],
            dtype=np.float32)
        
        
        self.train_adj_rhn = np.array(
            [add_global_node(padzero(np.array(sample['adjacency_matrix_rhn']), True, maxsize=self.maxsize), True) for sample
             in self.__train_dataset],
            dtype=np.float32)
        
        self.train_features_cnn = np.array(
            [add_global_node(padzero(np.array(sample['operations_cnn']), False, maxsize=self.maxsize), False) for sample in
             self.__train_dataset],
            dtype=np.float32)
        
        self.train_features_rhn = np.array(
            [add_global_node(padzero(np.array(sample['operations_rhn']), False, maxsize=self.maxsize), False) for sample in
             self.__train_dataset],
            dtype=np.float32)

        self.train_Y = np.array([sample['metrics'] for sample in self.__train_dataset], dtype=np.float32)
    
    # update GCN with the new ovserved points
    # just run train() as above
    def retrain_NN(self):
        self.__train_dataset = self.__dataset
        print('len_data')
        print(len(self.__train_dataset))
        neural_net = nn.NeuralNet(dataset=self.__train_dataset, val_dataset=self.__valset, ifPretrained=self.ifPretrain,
                                  maxsize=self.maxsize)
        neural_net.train(num_epoch=self.num_epoch, lr=self.lr, selected_loss=self.loss,
                         ifsigmoid=self.ifTransformSigmoid)
        self.gcn = neural_net.gcn

    # update GCN with the new observed points
    # just run train() of bayesian_sigmoid_regression
    def retrain_LR(self):
        """
        retrain bo regressor with updated dataset
        """
        start = time.time()
        train_features = self.extract_features(self.train_adj_cnn, self.train_features_cnn, self.train_adj_rhn, self.train_features_rhn)
        lm_dataset = (train_features, self.train_Y)
        # Train and predict with linear_regressor
        linear_regressor = lm.LinearRegressor(lm_dataset, intercept=False, ifTransformSigmoid=self.ifTransformSigmoid)
        linear_regressor.train()

        time_ = time.time()
        print(f"retrain lr time:{time_ - start}")

    def extract_features(self, adj_cnn, features_cnn, adj_rhn, features_rhn):
        adj_cnn = torch.Tensor(adj_cnn).to(device)
        features_cnn = torch.Tensor(features_cnn).to(device)
        adj_rhn = torch.Tensor(adj_rhn).to(device)
        features_rhn = torch.Tensor(features_rhn).to(device)
        with torch.no_grad():
            self.gcn.eval()
            embeddings = self.gcn(features_cnn, adj_cnn, features_rhn, adj_rhn, extract_embedding=True)
        return embeddings

    def get_ei(self, train_Y, prediction, hi_ci):
        if self.ifTransformSigmoid:
            train_Y = np.log(train_Y / (1 - train_Y))
        # sigma, standard deviation
        sig = abs((hi_ci - prediction) / 2)
        # seems to be backward
        if self.ifFindMax:
            gamma = (prediction - np.max(train_Y)) / sig
        else:
            gamma = (np.max(train_Y) - prediction) / sig
        # aquisition function defined in paper
        ei = sig * (gamma * stats.norm.cdf(gamma) + 1 * stats.norm.pdf(gamma))
        return ei, sig, gamma

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # prediction = pred
    def get_ucb(self, train_Y, prediction, hi_ci):
        gamma = np.pi / 8
        alpha = 4 - 2 * np.sqrt(2)
        beta = -np.log(np.sqrt(2) + 1)
        sig = abs((hi_ci - prediction) / 2)
        E = self.sigmoid(prediction / np.sqrt(1 + gamma * sig ** 2))
        std = np.sqrt(self.sigmoid((alpha * (prediction + beta)) / np.sqrt(1 + gamma * alpha**2 * sig**2)) - E**2)
        # aquisition function defined in paper
        ucb = E + 0.5 * std
        return ucb, std

    def select_multiple(self, new_domain, cap=5):
        """
        Identify multiple points. New_domain is to support sample algos such as RL and EA
        """
        # Rank order by ucb
        pred_true, ucb, sig = self.get_prediction(new_domain)
        
        ucb_order = np.argsort(-1 * ucb, axis=0)
        select_indices = [ucb_order[0, 0]]
        for candidate in ucb_order[1:, 0]:
            if ucb[candidate, 0] > 0:
                select_indices.append(candidate)
            if len(select_indices) == cap:  # Number of points to select
                break

        if len(select_indices) < cap:
            # If not enough good points, append with exploration
            sig_order = np.argsort(-sig, axis=0)
            add_indices = sig_order[:(cap - len(select_indices)), 0].tolist()
            select_indices.extend(add_indices)
        pred_acc = [pred_true[i] for i in select_indices]
        newdataset = [new_domain[i] for i in select_indices]
        logging.info("selected indices:{}".format(str(select_indices)))
        logging.info("length of selects:{}".format(str(len(select_indices))))
        return newdataset, pred_acc, select_indices

    def select_multiple_unique(self, new_domain, trained_models, cap=5):
        """
        Identify multiple points. New_domain is to support sample algos such as RL and EA
        """
        # Rank order by ucb
        pred_true, ucb, sig = self.get_prediction(new_domain)
        # pred_true, ucb, sig = get_prediction(new_domain)

        ucb_order = np.argsort(-1 * ucb, axis=0) # sort the ucb scores of the archs
        select_indices = [ucb_order[0, 0]] # get indices of ordered ucb scores
        for candidate in ucb_order[1:, 0]: # iterate over ucb_ordered values, until cap is reached: cap is stopping criterium, which means number of max values, which we will train
            if ucb[candidate, 0] > 0:
                data_point = new_domain[candidate]
                adj_cnn, ops_cnn, adj_rhn, ops_rhn = data_point['adjacency_matrix_cnn'], data_point['operations_cnn'], data_point['adjacency_matrix_rhn'], data_point['operations_rhn']
                genohash = str(hash(str(transform_Genotype(adj_cnn, ops_cnn, adj_rhn, ops_rhn))))
                if genohash not in trained_models:
                    select_indices.append(candidate)
            if len(select_indices) == cap:  # Number of points to select
                break

        sig_order = np.argsort(-sig, axis=0)
        add_indices = sig_order[:, 0].tolist()
        # If not enough good points, append with exploration
        for i in range(len(add_indices)):
            if len(select_indices) < cap:
                data_point = new_domain[add_indices[i]]
                adj_cnn, ops_cnn, adj_rhn, ops_rhn = data_point['adjacency_matrix_cnn'], data_point['operations_cnn'], data_point['adjacency_matrix_rhn'], data_point['operations_rhn']
                genohash = str(hash(str(transform_Genotype(adj_cnn, ops_cnn, adj_rhn, ops_rhn))))
                if genohash not in trained_models:
                    select_indices.append(add_indices[i])
            else:
                break
        pred_acc = [pred_true[i] for i in select_indices]
        newdataset = [new_domain[i] for i in select_indices]
        logging.info("selected indices:{}".format(str(select_indices)))
        logging.info("length of selects:{}".format(str(len(select_indices))))
        return newdataset, pred_acc, select_indices


    # new_dataset = trained_arch_list # saved properties like wie adj. matrix, accuracy, genotpyes of 100 init_samples
    def update_data(self, new_dataset):
        self.__dataset = new_dataset
        self.__train_dataset = self.__dataset
        
        self.train_adj_cnn = np.array( # for-loop iterates 100 times through trained_arch_list and always returns 1st train_arch
            [add_global_node(padzero(np.array(sample['adjacency_matrix_cnn']), True, maxsize=self.maxsize), True) for sample
             in self.__train_dataset],
            dtype=np.float32) 
        
        self.train_features_cnn = np.array(
            [add_global_node(padzero(np.array(sample['operations_cnn']), False, maxsize=self.maxsize), False) for sample in
             self.__train_dataset],
            dtype=np.float32)
        
        self.train_adj_rhn = np.array( # for-loop geht 100 mal durch trained_arch_list und gibt jedesmal 1e train_arch raus
            [add_global_node(padzero(np.array(sample['adjacency_matrix_rhn']), True, maxsize=self.maxsize), True) for sample
             in self.__train_dataset],
            dtype=np.float32) 
        
        self.train_features_rhn = np.array(
            [add_global_node(padzero(np.array(sample['operations_rhn']), False, maxsize=self.maxsize), False) for sample in
             self.__train_dataset],
            dtype=np.float32)
        
        
        self.train_Y = np.array([sample['metrics'] for sample in self.__train_dataset], dtype=np.float32)

    def get_prediction(self, new_domain, detail=True):
        
        domain_adj_cnn = np.array(
            [add_global_node(padzero(np.array(sample['adjacency_matrix_cnn']), True, maxsize=self.maxsize), True) for sample
             in new_domain],
            dtype=np.float32)
        domain_feature_cnn = np.array(
            [add_global_node(padzero(np.array(sample['operations_cnn']), False, maxsize=self.maxsize), False) for sample in
             new_domain],
            dtype=np.float32)
        
        domain_adj_rhn = np.array(
            [add_global_node(padzero(np.array(sample['adjacency_matrix_rhn']), True, maxsize=self.maxsize), True) for sample
             in new_domain],
            dtype=np.float32)
        domain_feature_rhn = np.array(
            [add_global_node(padzero(np.array(sample['operations_rhn']), False, maxsize=self.maxsize), False) for sample in
             new_domain],
            dtype=np.float32)
        
        train_features = self.extract_features(self.train_adj_cnn, self.train_features_cnn, self.train_adj_rhn, self.train_features_rhn)
        domain_features = self.extract_features(domain_adj_cnn, domain_feature_cnn, domain_adj_rhn, domain_feature_rhn)
        lm_dataset = (train_features, self.train_Y)
        linear_regressor = lm.LinearRegressor(lm_dataset, intercept=False, ifTransformSigmoid=self.ifTransformSigmoid)
        linear_regressor.train()
        pred, hi_ci, lo_ci, pred_true = linear_regressor.predict(domain_features)

        train_Y = self.train_Y
        ucb, sig = self.get_ucb(train_Y, pred, hi_ci)
        if detail:
            return pred_true, ucb, sig
        else:
            return pred_true, ucb

    def get_dataset(self):
        return self.__dataset

    def get_train(self):
        return self.__train_dataset

    def get_val(self):
        return self.__val_dataset
