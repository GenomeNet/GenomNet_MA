#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 17:30:39 2021

@author: amadeu
"""

gcn_epochs=args.gcn_epochs#100,
gcn_lr=args.gcn_lr
loss_num=args.loss_num
generate_num=args.generate_num#100,
iterations=args.iterations
bo_sample_num=args.bo_sample_num#100,
sample_method=args.sample_method
if_init_samples=args.if_init_samples
init_num=args.init_num#100,
save = args.save
save_dir = args.save_dir
train_supernet_epochs=args.train_supernet_epochs
data_path=os.path.join(local_data_dir, 'data')
super_batch_size=args.super_batch_size
sub_batch_size=args.sub_batch_size
cnn_lr=args.cnn_lr
cnn_weight_decay=args.cnn_weight_decay
rhn_lr=args.rhn_lr
rhn_weight_decay=args.rhn_weight_decay
momentum=args.momentum
report_freq=args.report_freq
epochs=args.epochs
init_channels=args.init_channels#36
layers= args.layers
drop_path_prob=args.drop_path_prob
seed=0
grad_clip=args.clip
parallel=False
mode=args.mode
train_directory = args.train_directory
valid_directory = args.valid_directory
test_directory = args.test_directory
train_input_directory = args.train_input_directory
train_target_directory = args.train_target_directory
valid_input_directory = args.valid_input_directory
valid_target_directory = args.valid_target_directory
test_input_directory = args.test_input_directory
test_target_directory = args.test_target_directory
task = args.task
num_files = args.num_files
seq_len = args.seq_size
num_steps = args.num_steps
next_character_prediction=args.next_character_prediction
dropouth = args.dropouth
dropoutx = args.dropoutx
one_clip = args.one_clip
clip = args.clip
conv_clip = args.conv_clip
rhn_clip = args.rhn_clip 



import time
import numpy as np

import BO_tools.neural_net_gcn as nn
import BO_tools.linear_regressor as lm

import scipy.stats as stats
from predictors.utils.gcn_utils import padzero, add_global_node
from opendomain_utils.loss_function import weighted_exp, weighted_linear, weighted_log
from opendomain_utils.transform_genotype import transform_Genotype
import torch
import torch.nn.functional as F
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



num_epoch = 3

dataset = trained_arch_list
num_epoch=gcn_epochs
lr=gcn_lr
lossnum=loss_num
ifPretrain=False
ifTransformSigmoid=True
ifFindMax=True
maxsize=7
__dataset = trained_arch_list
lossList = [torch.nn.MSELoss(), weighted_log, weighted_linear, weighted_exp]
loss = lossList[lossnum]

class Optimizer(object):

    def __init__(self, dataset, val_set=None, ifPretrain=False, ifTransformSigmoid=True, ifFindMax=True,
                 lr=0.001, train_epoch=1, lossnum=3, maxsize=7):
        """Initialization of Optimizer object
        Keyword arguments:
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        what is architecture
        """
        # beim initialisieren wird nur funktion process_data() ausgeführt
        lossList = [torch.nn.MSELoss(), weighted_log, weighted_linear, weighted_exp]
        __dataset = dataset
        __valset = None
        process_data() # trained_arch_list wird eben als richtiges A und X bildet und abgespeichert
        ifPretrain = ifPretrain
        ifTransformSigmoid = ifTransformSigmoid
        ifFindMax = ifFindMax
        lr = lr
        num_epoch = 2
        loss = lossList[lossnum]

    def train(self):
        """ Using the stored dataset and architecture, trains the neural net to
        perform feature extraction, and the linear regressor to perform prediction
        and confidence interval computation.
        """
        start = time.time()
        __train_dataset = __dataset

        # wird einfach nur initialisiert, kein train oder so wird ausgeführt
        neural_net = nn.NeuralNet(dataset=__train_dataset, val_dataset=__valset, ifPretrained=ifPretrain,
                                  maxsize=maxsize) # oben wurde neural_net_gcn als nn importiert
        # neural_net = nn.NeuralNet(dataset=__train_dataset, val_dataset=__valset, ifPretrained=ifPretrain, maxsize=maxsize)
        
        # num_epoch=num_epoch, lr=lr, selected_loss=loss, ifsigmoid=ifTransformSigmoid
        # er führt train aus, wo GCN erstmal initialisiert wird und dann aber direkt über num_epoch trainiert wird. Das training
        # läuft eigentl wie immer, nur dass jetzt eben adj und feat als input
        #neural_net.train(num_epoch=self.num_epoch, lr=self.lr, selected_loss=self.loss,
        #                 ifsigmoid=self.ifTransformSigmoid)
        neural_net.train(num_epoch= num_epoch, lr= lr, selected_loss=loss, ifsigmoid= ifTransformSigmoid)
        
        gcn = neural_net.gcn
        # macht einfach nur embedding bei GCN -> also nicht final outpout vom GCN, sondern embedding output
        # aber haben jetzt eben unser [4, 128] also concated embedding, anstatt  [4,64]
        # den brauch ich dann nämlich 
        
        train_adj_cnn = np.array(
            [add_global_node(padzero(np.array(sample['adjacency_matrix_cnn']), True, maxsize=maxsize), True) for sample
             in __train_dataset],
            dtype=np.float32)
        
        train_adj_rhn = np.array(
            [add_global_node(padzero(np.array(sample['adjacency_matrix_rhn']), True, maxsize=maxsize), True) for sample
             in __train_dataset],
            dtype=np.float32)
        
        train_features_cnn = np.array(
            [add_global_node(padzero(np.array(sample['operations_cnn']), False, maxsize=maxsize), False) for sample in
             __train_dataset],
            dtype=np.float32)
        
        train_features_rhn = np.array(
            [add_global_node(padzero(np.array(sample['operations_rhn']), False, maxsize=maxsize), False) for sample in
             __train_dataset],
            dtype=np.float32)
        
    
        train_Y = np.array([sample['metrics'] for sample in __train_dataset], dtype=np.float32)
        
        
        train_features = extract_features(gcn, train_adj_cnn, train_features_cnn, train_adj_rhn, train_features_rhn)
        lm_dataset = (train_features, train_Y)
        # er initalisiert erstmal mit den Daten, macht also im Endeffekt noch nichts
        linear_regressor = lm.LinearRegressor(lm_dataset, intercept=False, ifTransformSigmoid=ifTransformSigmoid)
        
        # jetzt trainiert er das initialisierte linear_regressor model
        linear_regressor.train()
        time_ = time.time()
        print(f"train gcn time:{start - time_}")

   
    
    # update GCN with the new ovserved points
    # wobei er einfach nur wieder train() von oben ausführt
    def retrain_NN(self):
        self.__train_dataset = self.__dataset
        neural_net = nn.NeuralNet(dataset=self.__train_dataset, val_dataset=self.__valset, ifPretrained=self.ifPretrain,
                                  maxsize=self.maxsize)
        neural_net.train(num_epoch=self.num_epoch, lr=self.lr, selected_loss=self.loss,
                         ifsigmoid=self.ifTransformSigmoid)
        self.gcn = neural_net.gcn

    # update GCN with the new ovserved points
    # wobei er einfach nur wieder train() von bayesian_sigmoid_regression ausführt
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

    def extract_features(gcn, adj_cnn, features_cnn, adj_rhn, features_rhn):
        adj_cnn = torch.Tensor(adj_cnn).to(device)
        features_cnn = torch.Tensor(features_cnn).to(device)
        adj_rhn = torch.Tensor(adj_rhn).to(device)
        features_rhn = torch.Tensor(features_rhn).to(device)
        with torch.no_grad():
            gcn.eval()
            embeddings = gcn(features_cnn, adj_cnn, features_rhn, adj_rhn, extract_embedding=True)
        return embeddings
    
     # def extract_features(gcn, adj_cnn, features_cnn, adj_rhn, features_rhn):
     #   adj_cnn = torch.Tensor(adj_cnn).to(device)
     #   features_cnn = torch.Tensor(features_cnn).to(device)
     #   with torch.no_grad():
     #       gcn.eval()
     #       embeddings = gcn(features_cnn, adj_cnn, features_rhn, adj_rhn, extract_embedding=True)
     #   return embeddings

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

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # prediction = pred
    def get_ucb(sigmoid, train_Y, prediction, hi_ci):
        gamma = np.pi / 8
        alpha = 4 - 2 * np.sqrt(2)
        beta = -np.log(np.sqrt(2) + 1)
        sig = abs((hi_ci - prediction) / 2)
        E = sigmoid(prediction / np.sqrt(1 + gamma * sig**2))
        std = np.sqrt(sigmoid((alpha * (prediction + beta)) / np.sqrt(1 + gamma * alpha**2 * sig**2)) - E**2)
        # aquisition function defined in paper
        ucb = E + 0.5 * std
        return ucb, std

  

    def select_multiple_unique(self, new_domain, trained_models, cap=5):
        """
        Identify multiple points. New_domain is to support sample algos such as RL and EA
        """
        # Rank order by ucb
        pred_true, ucb, sig = get_prediction(new_domain, maxsize, train_adj_cnn, train_features_cnn, train_adj_rhn, train_features_rhn, train_Y, detail=True)
        # pred_true, ucb, sig = get_prediction(new_domain)

        # select the indices with highest ucb-score
        ucb_order = np.argsort(-1 * ucb, axis=0) # sort the ucb scores of the archs with indices: [0][6][7] etc.: it is ordered in ascending order, so the highest are at the top
        select_indices = [ucb_order[0, 0]] # get the highest index of ucb
        for candidate in ucb_order[1:, 0]: # iterate over ucb_ordered values and append all candidates until cap is reached: cap is stopping criterium, which means number of max values, which we will train
            # ucb_order[1:, 0]: alle ucb-scores aus das erste (erste ist select_indices)
            # candidate = 6
            if ucb[candidate, 0] > 0: # greift auf den ucb-score (auf 6tes element)
                data_point = new_domain[candidate] # greift auf 6tes element
                adj_cnn, ops_cnn, adj_rhn, ops_rhn = data_point['adjacency_matrix_cnn'], data_point['operations_cnn'], data_point['adjacency_matrix_rhn'], data_point['operations_rhn']
                genohash = str(hash(str(transform_Genotype(adj_cnn, ops_cnn, adj_rhn, ops_rhn))))
                if genohash not in trained_models: # add indices, only if genotype is not existing
                    select_indices.append(candidate)
            if len(select_indices) == cap:  # Number of points to select
                break

        sig_order = np.argsort(-sig, axis=0)
        add_indices = sig_order[:, 0].tolist()
        # If not enough good points, append with exploration (because, all genohash with high ucb-score are already existing and sigmoid-scores will give us different genotypes/genohashs)
        for i in range(len(add_indices)):
            if len(select_indices) < cap:
                data_point = new_domain[add_indices[i]]
                adj_cnn, ops_cnn, adj_rhn, ops_rhn = data_point['adjacency_matrix_cnn'], data_point['operations_cnn'], data_point['adjacency_matrix_rhn'], data_point['operations_rhn']
                genohash = str(hash(str(transform_Genotype(adj_cnn, ops_cnn, adj_rhn, ops_rhn))))
                if genohash not in trained_models: # add indices, only if genotype is not existing
                    select_indices.append(add_indices[i])
            else:
                break
        # define the pred_y of selected indices
        pred_acc = [pred_true[i] for i in select_indices]
        # define the adjs and ops of selected indices

        newdataset = [new_domain[i] for i in select_indices]
        logging.info("selected indices:{}".format(str(select_indices)))
        logging.info("length of selects:{}".format(str(len(select_indices))))
        return newdataset, pred_acc, select_indices


    # new_dataset = trained_arch_list # abgespeicherte eigenschaften wie adj. matrix, accuracy, genotpyes der 100 init_samples
    def update_data(self, new_dataset):
        self.__dataset = new_dataset
        self.__train_dataset = self.__dataset
        # samples = []
        # for sample in __train_dataset:
        #    samples.append(sample)
        
        self.train_adj_cnn = np.array( # for-loop geht 100 mal durch trained_arch_list und gibt jedesmal 1e train_arch raus
            [add_global_node(padzero(np.array(sample['adjacency_matrix_cnn']), True, maxsize=self.maxsize), True) for sample
             in self.__train_dataset],
            dtype=np.float32) 
        
        #train_adj = np.array( #
        #    [add_global_node(padzero(np.array(sample['adjacency_matrix']), True, maxsize=maxsize), True) for sample
        #     in __train_dataset], dtype=np.float32)
        
        
        self.train_features_cnn = np.array(
            [add_global_node(padzero(np.array(sample['operations_cnn']), False, maxsize=self.maxsize), False) for sample in
             self.__train_dataset],
            dtype=np.float32)
        
        #train_features = np.array(
        #    [add_global_node(padzero(np.array(sample['operations']), False, maxsize=maxsize), False) for sample in
        #     __train_dataset],
        #    dtype=np.float32)
        
        self.train_adj_rhn = np.array( # for-loop geht 100 mal durch trained_arch_list und gibt jedesmal 1e train_arch raus
            [add_global_node(padzero(np.array(sample['adjacency_matrix_rhn']), True, maxsize=self.maxsize), True) for sample
             in self.__train_dataset],
            dtype=np.float32) 
        
        #train_adj = np.array( #
        #    [add_global_node(padzero(np.array(sample['adjacency_matrix']), True, maxsize=maxsize), True) for sample
        #     in __train_dataset], dtype=np.float32)
        
        
        self.train_features_rhn = np.array(
            [add_global_node(padzero(np.array(sample['operations_rhn']), False, maxsize=self.maxsize), False) for sample in
             self.__train_dataset],
            dtype=np.float32)
        
        #train_features = np.array(
        #    [add_global_node(padzero(np.array(sample['operations']), False, maxsize=maxsize), False) for sample in
        #     __train_dataset],
        #    dtype=np.float32)
        
        
        self.train_Y = np.array([sample['metrics'] for sample in self.__train_dataset], dtype=np.float32)


    def get_prediction(new_domain, maxsize, train_adj_cnn, train_features_cnn, train_adj_rhn, train_features_rhn, train_Y, detail=True):
        
        domain_adj_cnn = np.array(
            [add_global_node(padzero(np.array(sample['adjacency_matrix_cnn']), True, maxsize=maxsize), True) for sample
             in new_domain],
            dtype=np.float32)
        domain_feature_cnn = np.array(
            [add_global_node(padzero(np.array(sample['operations_cnn']), False, maxsize=maxsize), False) for sample in
             new_domain],
            dtype=np.float32)
        
        domain_adj_rhn = np.array(
            [add_global_node(padzero(np.array(sample['adjacency_matrix_rhn']), True, maxsize=maxsize), True) for sample
             in new_domain],
            dtype=np.float32)
        domain_feature_rhn = np.array(
            [add_global_node(padzero(np.array(sample['operations_rhn']), False, maxsize=maxsize), False) for sample in
             new_domain],
            dtype=np.float32)
        
        
        train_features = extract_features(gcn,train_adj_cnn, train_features_cnn, train_adj_rhn, train_features_rhn)
        domain_features = extract_features(gcn,domain_adj_cnn, domain_feature_cnn, domain_adj_rhn, domain_feature_rhn)
        # train_features = extract_features(train_adj, train_features)
        # domain_features = extract_features(domain_adj, domain_feature)
        lm_dataset = (train_features, train_Y)
        linear_regressor = lm.LinearRegressor(lm_dataset, intercept=False, ifTransformSigmoid=ifTransformSigmoid)
        linear_regressor.train()
        pred, hi_ci, lo_ci, pred_true = linear_regressor.predict(domain_features)

        train_Y = train_Y
        ucb, sig = get_ucb(sigmoid, train_Y, pred, hi_ci)
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






 def evaluate(model, mask, valid_loader, num_steps, criterion, task, report_freq):
        objs = utils.AvgrageMeter()
        #top1 = utils.AvgrageMeter()
        #top2 = utils.AvgrageMeter()
        model.eval()
             
        total_loss = 0
        labels = []
        predictions = []
        
        scores = nn.Softmax()
        
        for step, (input, target) in enumerate(valid_loader):

            if step > num_steps:
                break
            
            # input = input.transpose(1,2).float()
            #print(input.shape)
            input = input.to(device).float()
            batch_size = input.size(0)
    
            target = target.to(device)
            #target = torch.max(target, 1)[1]
            hidden = model.init_hidden(batch_size)#.to(device)  
    
            with torch.no_grad():
                # print(target)
                logits, hidden = model(input, hidden, mask)
                #print(logits)
                loss = criterion(logits, target)
    
            # prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
            
            objs.update(loss.data, batch_size)
            labels.append(target.detach().cpu().numpy())
            if task == "next_character_prediction":
                predictions.append(scores(logits).detach().cpu().numpy())
            else:#if args.task == "TF_bindings"::
                predictions.append(logits.detach().cpu().numpy())
    
            if step % report_freq == 0:
                #logging.info('| step {:3d} | val obj {:5.2f} | '
                #    'val acc {:8.2f}'.format(step,
                #                               objs.avg, top1.avg))
                logging.info('| step {:3d} | val obj {:5.2f}'.format(step, objs.avg))


        return labels, predictions, objs.avg.detach().cpu().numpy() # t

