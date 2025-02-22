#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:25:50 2021

@author: amadeu
"""

import numpy as np
from BO_tools.alpha_beta import fit as get_alpha_beta
import torch


# dataset, intercept, ifTransformSigmoid = lm_dataset, False, ifTransformSigmoid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class LinearRegressor():

    def __init__(self, dataset, intercept=True, ifTransformSigmoid=True):
        """Initialization of Optimizer object
        Keyword arguments:
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        self.__dataset = dataset
        self.__intercept = intercept
        self.ifTransformSigmoid = ifTransformSigmoid

    # train() is only done to determine "m", "beta" and "XX_inv", which are then used for predict
    # in the end this does nothing more than determine posterior_mean and posterior_variance etc. with Maximum-Likelihood
    # 
    def train(self):
        dataset = self.__dataset
        train_X = dataset[0] # just X, because second element is y; shape [4,128]
        # (output of gcn embedding layer)
        train_X = train_X.cpu().data.numpy()
        train_Y = dataset[1]
        if self.ifTransformSigmoid:
            train_Y = np.log(train_Y / (1 - train_Y))
        # xx_inv stands for K inverse, paper used sum instead of lstsq, paper has beta and alpha as hyper parameter
        # beta stands for m, paper has beta as hyper parameter, 64*1
        # imported above as fit
        m, XX_inv, beta = get_alpha_beta(train_X, train_Y) # m_N, S_N, beta
        self.__XX_inv = XX_inv
        self.__m = m
        self.beta = beta

    # what is intercept?
    def predict(self, test_X, fc=None):
        
        test_X = test_X.cpu().data.numpy()
        XX_inv = self.__XX_inv
        m = self.__m
        s = []
        for row in range(test_X.shape[0]):
            x = test_X[row]
            # x.shape: [4,128] (output of gcn embedding layer)
            s.append((1 / self.beta + np.dot(np.dot(x, XX_inv), x.T)) ** 0.5)

        s = np.reshape(np.asarray(s), (test_X.shape[0], 1))
        
        if fc:
            test_X = torch.from_numpy(test_X).to(device)
            test_pred = fc(test_X)
            test_pred = test_pred.cpu().data.numpy()
        else:
            test_pred = np.dot(test_X, m)
        if self.ifTransformSigmoid:
            test_pred_true = 1 / (1 + np.exp(-test_pred))
        else:
            test_pred_true = test_pred
        test_pred = np.reshape(test_pred, (test_X.shape[0], 1))
        test_pred_true = np.reshape(test_pred_true, (test_X.shape[0], 1))
        hi_ci = test_pred + 2 * s
        lo_ci = test_pred - 2 * s
        return test_pred, hi_ci, lo_ci, test_pred_true
