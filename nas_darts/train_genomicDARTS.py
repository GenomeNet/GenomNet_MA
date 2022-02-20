#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:24:24 2021

@author: amadeu
"""

import argparse
import os, sys
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn

from darts_tools.architect import Architect

import torch.utils

import copy
import nas_utils.model_search as one_shot_model

from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from generalNAS_tools.train_and_validate import train, infer

from generalNAS_tools import utils

from generalNAS_tools.utils import overall_acc, overall_f1


parser = argparse.ArgumentParser(description='DARTS for genomic Data')

parser.add_argument('--task', type=str, default='TF_bindings', help='defines the task: next_character_prediction (not fully implemented!) or TF_bindings (default)')

parser.add_argument('--seed', type=int, default=3, help='random seed')

parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--cnn_lr', type=float, default=0.025, help='learning rate for CNN part')
parser.add_argument('--cnn_weight_decay', type=float, default=3e-4, help='weight decay for CNN part')
parser.add_argument('--rhn_lr', type=float, default=2, help='learning rate for RHN part')
parser.add_argument('--rhn_weight_decay', type=float, default=5e-7, help='weight decay for RHN part')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')

parser.add_argument('--one_clip', dest='one_clip', action='store_true', help='use --clip value for both cnn and rhn gradient clipping (default).')
parser.add_argument('--no-one_clip', dest='one_clip', action='store_false', help='disable one_clip: use --conv_clip and --rhn_clip')
parser.set_defaults(one_clip=True)
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping when --one-clip is given')
parser.add_argument('--conv_clip', type=float, default=5, help='gradient clipping of convs')
parser.add_argument('--rhn_clip', type=float, default=0.25, help='gradient clipping of lstms')

parser.add_argument('--dropouth', type=float, default=0.25, help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75, help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--beta', type=float, default=1e-3, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')

parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')

parser.add_argument('--num_steps', type=int, default=2000, help='number of iterations per epoch')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')

parser.add_argument('--steps', type=int, default=4, help='total number of Nodes')

parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='batch size')



parser.add_argument('--train_directory', type=str, default='data/deepsea_train/train.mat', help='file (TF_bindings) or directory (next_character_prediction) of training data')
parser.add_argument('--valid_directory', type=str, default='data/deepsea_train/valid.mat', help='file (TF_bindings) or directory (next_character_prediction) of validation data')
parser.add_argument('--test_directory', type=str, default='data/deepsea_train/test.mat', help='file (TF_bindings) or directory (next_character_prediction) of test data')

parser.add_argument('--seq_size', type=int, default=1000, help='input sequence size')

parser.add_argument('--next_character_predict_character', dest='next_character_prediction', action='store_true', help='only for --task=next_character_prediction: predict single character')
parser.add_argument('--next_character_predict_sequence', dest='next_character_prediction', action='store_false', help='only for --task=next_character_prediction: predict entire sequence, shifted by one character, using causal convolutions')
parser.set_defaults(next_character_prediction=True)

parser.add_argument('--num_files', type=int, default=3, help='number of files for training data (for --task=next_character_prediction)')


parser.add_argument('--report_freq', type=int, default=1, metavar='N', help='report interval')

parser.add_argument('--report_validation', type=int, default=1, help='validation report period; default 1 (every epoch)')
parser.add_argument('--validation', dest='validation', action='store_true', help='do validation (default)')
parser.add_argument('--no-validation', dest='validation', action='store_false', help='no validation; disables --validation')
parser.set_defaults(validation=True)

parser.add_argument('--save', type=str,  default='search', help='file name postfix to save the labels and predicitons')
parser.add_argument('--save_dir', type=str,  default='test_search', help='path to save the labels and predicitons')

args = parser.parse_args()


utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
#logging.getLogger().addHandler(fh)

logging = logging.getLogger(__name__)


def main():

    torch.manual_seed(args.seed)
      
    logging.info("args = %s", args)
           
    if args.task == "next_character_prediction":
        
        import generalNAS_tools.data_preprocessing_new as dp

        train_queue, valid_queue, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
                seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)
      
        _, test_queue, _ = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.test_directory, num_files=args.num_files,
                seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)
        
        criterion = nn.CrossEntropyLoss().to(device)
        
    if args.task == 'TF_bindings':
        
        import generalNAS_tools.data_preprocessing_TF as dp
        
        train_queue, valid_queue, test_queue = dp.data_preprocessing(args.train_directory, args.valid_directory, args.test_directory, args.batch_size)
        
        criterion = nn.BCELoss().to(device)
        
        num_classes = 919
    
        
    # build Network
    
    # initialize switches
    # in DARTS they should always be True for all operations, because we keep all operations and edges during
    # the search process
    switches_cnn = [] 
    for i in range(14):
        switches_cnn.append([True for j in range(len(PRIMITIVES_cnn))])
        
    switches_normal_cnn = copy.deepcopy(switches_cnn)
    switches_reduce_cnn = copy.deepcopy(switches_cnn)
    
    # get switches for RNN part
    switches_rnn = [] 
    for i in range(36):
        switches_rnn.append([True for j in range(len(PRIMITIVES_rnn))]) 
    switches_rnn = copy.deepcopy(switches_rnn)
    
    
    # initialize alpha weights
    k_cnn = sum(1 for i in range(args.steps) for n in range(2+i))
          
    num_ops_cnn = sum(switches_normal_cnn[0])
           
    alphas_normal = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_cnn, num_ops_cnn))) # was: k_cnn, num_ops_cnn
    alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_cnn, num_ops_cnn))) # was: k_cnn, num_ops_cnn
           
    k_rhn = sum(i for i in range(1, rnn_steps+1))             
    num_ops_rhn = sum(switches_rnn[0])
           
    alphas_rnn = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_rhn, num_ops_rhn)))
    
    multiplier, stem_multiplier = 4,3
    

    model = one_shot_model.RNNModelSearch(args.seq_size, args.dropouth, args.dropoutx,
                              args.init_channels, num_classes, args.layers, args.steps, multiplier, stem_multiplier,  
                              True, 0.2, None, args.task, 
                              switches_normal_cnn, switches_reduce_cnn, switches_rnn, 0.0, alphas_normal, alphas_reduce, alphas_rnn) 
                                  
    model = model.to(device)
    
    
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    
    conv = []
    rhn = []
    for name, param in model.named_parameters():
       if 'rnns' in name:
           rhn.append(param)
       else:
           conv.append(param)
    
    optimizer = torch.optim.SGD([{'params':conv}, {'params':rhn}], lr=args.cnn_lr, weight_decay=args.cnn_weight_decay)
    optimizer.param_groups[0]['lr'] = args.cnn_lr
    optimizer.param_groups[0]['weight_decay'] = args.cnn_weight_decay
    optimizer.param_groups[0]['momentum'] = args.momentum
    optimizer.param_groups[1]['lr'] = args.rhn_lr
    optimizer.param_groups[1]['weight_decay'] = args.rhn_weight_decay
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    
    sm_dim = -1
    epochs = args.epochs
    scale_factor = 0.2
    
    architect = Architect(model, criterion, args)

    train_losses = []
    valid_losses = []
    
    train_acc = []
    
    valid_acc = []
    
    time_per_epoch = []
    
    clip_params = [args.conv_clip, args.rhn_clip, args.clip]
    
    eps_no_arch = 5
    
    all_labels_valid = []
    all_predictions_valid = []

    for epoch in range(epochs):
            train_start = time.strftime("%Y%m%d-%H%M")
   
            lr = scheduler.get_last_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()
            
            if epoch < eps_no_arch: 
                labels, predictions, train_loss = train(train_queue, valid_queue, model, rhn, conv, criterion, optimizer, None, architect, args.unrolled, lr, epoch, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, train_arch=False, pdarts=False, task=args.task)
            else:
                labels, predictions, train_loss = train(train_queue, valid_queue, model, rhn, conv, criterion, optimizer, None, architect, args.unrolled, lr, epoch, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, train_arch=True, pdarts=False, task=args.task)

            
            labels = np.concatenate(labels)
            predictions = np.concatenate(predictions)
            
            scheduler.step()

            
            if args.task == 'next_character_prediction':
                acc = overall_acc(labels, predictions, args.task)
                logging.info('| epoch {:3d} | train acc {:5.2f}'.format(epoch, acc))
                train_acc.append(acc)
            else: 
            
                f1 = overall_f1(labels, predictions, args.task)
                logging.info('| epoch {:3d} | train f1-score {:5.2f}'.format(epoch, f1))
                train_acc.append(f1)

     
            train_losses.append(train_loss)
            epoch_end = time.time()
            time_per_epoch.append(epoch_end)

            
            # validation
            if args.validation == True:
                if epoch % args.report_validation == 0:
                    labels, predictions, valid_loss = infer(valid_queue, model, criterion, args.batch_size, args.num_steps, args.report_freq, task=args.task)
                    
                    labels = np.concatenate(labels)
                    predictions = np.concatenate(predictions)
                    
                    valid_losses.append(valid_loss)
                    logging.info('| epoch {:3d} | valid loss {:5.2f}'.format(epoch, valid_loss))

            
                    if args.task == 'next_character_prediction':
                        acc = overall_acc(labels, predictions, args.task)
                        logging.info('| epoch {:3d} | valid acc {:5.2f}'.format(epoch, acc))
                        valid_acc.append(acc)
                    else:
                        f1 = overall_f1(labels, predictions, args.task)
                        logging.info('| epoch {:3d} | valid f1-score {:5.2f}'.format(epoch, f1))
                        valid_acc.append(f1)
            
          
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            time_per_epoch.append(epoch_end)
            
            genotype = model.genotype()
            logging.info(genotype) 
            
    
    all_labels_valid.append(labels)
    all_predictions_valid.append(predictions)
    
    
    genotype_file = 'darts_geno-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, genotype_file), genotype)
    
    trainloss_file = 'train_loss-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, trainloss_file), train_losses)
    
    acc_train_file = 'acc_train-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, acc_train_file), train_acc)

    time_file = 'time-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, time_file), time_per_epoch)

    # safe valid data
    validloss_file = 'valid_loss-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, validloss_file), valid_losses)

    acc_valid_file = 'acc_valid-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, acc_valid_file), valid_acc)
    
    
      

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
