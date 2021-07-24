#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 08:39:30 2021

@author: amadeu
"""

import argparse
import os, sys, glob
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
#from architect import Architect
import time

#import genotypes_rnn
#import genotypes_cnn
from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype

import gc

#import data
# import model_searchCNN as oneshot_model
import darts_tools.model_search as one_shot_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint 
from generalNAS_tools import utils

import generalNAS_tools.data_preprocessing_new as dp

from generalNAS_tools.train_and_validate import train, infer

import darts_tools.cnn_eval



parser = argparse.ArgumentParser(description='Evaluate final architecture found by PDARTS')
parser.add_argument('--data', type=str, default='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt', help='location of the data corpus')
parser.add_argument('--num_steps', type=int, default=2, help='number of iterations per epoch')
parser.add_argument('--train_directory', type=str, default='/home/amadeu/Downloads/genomicData/train', help='directory of training data')
parser.add_argument('--valid_directory', type=str, default='/home/amadeu/Downloads/genomicData/validation', help='directory of validation data')
parser.add_argument('--num_files', type=int, default=3, help='number of files for data')
parser.add_argument('--cnn_lr', type=float, default=0.025, help='learning rate for CNN part')
parser.add_argument('--cnn_weight_decay', type=float, default=3e-4, help='weight decay for CNN part')
parser.add_argument('--rhn_lr', type=float, default=2, help='learning rate for RHN part')
parser.add_argument('--rhn_weight_decay', type=float, default=5e-7, help='weight decay for RHN part')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--validation', type=bool, default=True)
parser.add_argument('--report_validation', type=int, default=2, help='validation epochs') 
parser.add_argument('--next_character_prediction', type=bool, default=True, help='task of model')
parser.add_argument('--one_clip', type=bool, default=True)
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--conv_clip', type=float, default=5, help='gradient clipping of convs')
parser.add_argument('--rhn_clip', type=float, default=0.25, help='gradient clipping of lstms')
parser.add_argument('--init_channels', type=int, default=8, help='num of init channels') # args.C, args.num_classes, args.layers, args.steps=4, args.multiplier=4, args.stem_multiplier=3
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--num_classes', type=int, default=4, help='num of output classes') # args.C, args.num_classes, args.layers, args.steps=4, args.multiplier=4, args.stem_multiplier=3
parser.add_argument('--steps', type=int, default=4, help='total number of Nodes')
parser.add_argument('--multiplier', type=int, default=4, help='multiplier')
parser.add_argument('--stem_multiplier', type=int, default=3, help='stem multiplier')
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
parser.add_argument('--num_runs', type=int, default=2, metavar='N',
                    help='number of runs')
parser.add_argument('--seq_len', type=int, default=20,
                    help='sequence length')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_wdecay', type=float, default=1e-3,
                    help='weight decay for the architecture encoding alpha')
parser.add_argument('--arch_lr', type=float, default=3e-3,
                    help='learning rate for the architecture encoding alpha')
parser.add_argument('--genotype_file', type=str, default='/home/amadeu/anaconda3/envs/EXPsearch-try-20210620-144357-pdarts_geno.npy', help='directory of final genotype')
parser.add_argument('--search', type=bool, default=False, help='which architecture to use')
#parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
#parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')
args = parser.parse_args()


args.save = '{}search-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))



def main():
    
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    # criterion = nn.BCELoss()
  
    if (args.task == "next_character_prediction"):
        import generalNAS_tools.data_preprocessing_new as dp
      
        train_queue, valid_queue, test_queue, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, test_directory = args.test_directory, num_files=args.num_files,
                  seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)

      
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
          
      
    if (args.task == "TF_bindings"):
      
        import generalNAS_tools.data_preprocessing_TF as dp
        
        train_queue, valid_queue, test_queue = dp.data_preprocessing(args.train_input_directory, args.valid_input_directory, args.test_input_directory, args.train_target_directory, args.valid_target_directory, args.test_target_direcotry, args.batch_size)
        
        criterion = nn.BCELoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-6, momentum=0.9)
    
    
    
    
    
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    #logging.getLogger().addHandler(fh)
    
    logging = logging.getLogger(__name__)
    
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    if args.continue_train:
        model = torch.load(os.path.join(args.save, 'model.pt'))
    else:      
        
        genotype = np.load(args.genotype_file, allow_pickle=True)
        # genotype = np.load('/home/amadeu/Desktop/GenomNet_MA/EXPsearch-try-20210626-091257-random_geno.npy', allow_pickle=True)
    
        # my_gene = genotype
        # his_gene = genotype
    
    
        model = one_shot_model.RNNModel(args.seq_len, args.dropouth, args.dropoutx,
                            args.init_channels, args.num_classes, args.layers, args.steps, args.multiplier, args.stem_multiplier,
                            args.search, args.drop_path_prob, genotype=genotype, task=args.task).to(device)
        
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    
    conv = []
    rhn = []
    for name, param in model.named_parameters():
        #print(name)
        #if 'stem' or 'preprocess' or 'conv' or 'bn' or 'fc' in name:
        if 'rnns' in name:
            #print(name)
            rhn.append(param)
            #elif 'decoder' in name:
        else:
            #print(name)
            conv.append(param)
            
    optimizer = torch.optim.SGD([{'params':conv}, {'params':rhn}], lr=args.cnn_lr, weight_decay=args.cnn_weight_decay)
    optimizer.param_groups[0]['lr'] = args.cnn_lr
    optimizer.param_groups[0]['weight_decay'] = args.cnn_weight_decay
    optimizer.param_groups[0]['momentum'] = args.momentum
    optimizer.param_groups[1]['lr'] = args.rhn_lr
    optimizer.param_groups[1]['weight_decay'] = args.rhn_weight_decay
            
    
            
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
             optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    
    
    optimizer_a=None
    
    clip_params = [args.conv_clip, args.rhn_clip, args.clip]
    
    
    #for run in range(args.num_runs):
        # run=0
        
    train_losses = []
    valid_losses = []
    
    train_acc = []
    #all_predictions_train = []
    valid_acc = []
    #all_predictions_valid = []
    time_per_epoch = []
    
    train_start = time.strftime("%Y%m%d-%H%M")

    for epoch in range(args.epochs):
        # epoch=1
                    
        epoch_start_time = time.time()
            
        lr = scheduler.get_last_lr()[0]
    
        labels, predictions, train_loss = train(train_object, valid_object, model, rhn, conv, criterion, optimizer, optimizer_a, lr, epoch, args.num_steps, clip_params, args.one_clip, args.report_freq, args.beta, train_arch=False)
            
        scheduler.step()
    
        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        
        if args.task == 'next_character_prediction':
                acc = overall_acc(labels, predictions, args.task)
                logging.info('| epoch {:3d} | train acc {:5.2f}'.format(epoch, acc))
                train_acc.append(acc)


        else:
                f1 = overall_f1(labels, predictions, args.task)
                logging.info('| epoch {:3d} | train f1-score {:5.2f}'.format(epoch, f1))
                train_acc.append(f1)

     
      
        #all_labels_train.append(labels)
        #all_predictions_train.append(predictions)
        train_losses.append(train_loss)
        epoch_end = time.time()
        time_per_epoch.append(epoch_end)
      
       
            
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'train_-epoch{}.pth'.format(epoch)) 
            
        # determine the validation loss in every 5th epoch 
        if args.validation == True:
                
            if epoch % args.report_validation == 0:
                    
                labels, predictions, valid_loss = infer(valid_object, model, criterion, args.batch_size, args.num_steps, args.report_freq)
                logging.info('Valid_acc %f', valid_acc)
               
                valid_losses.append(valid_loss)
                #all_labels_valid.append(labels)
                #all_predictions_valid.append(predictions)
            
                if args.task == 'next_character_prediction':
                    acc = overall_acc(labels, predictions, args.task)
                    logging.info('| epoch {:3d} | valid acc {:5.2f}'.format(epoch, acc))
                    valid_acc.append(acc)

                else:
                    f1 = overall_f1(labels, predictions, args.task)
                    logging.info('| epoch {:3d} | valid f1-score {:5.2f}'.format(epoch, f1))
                    valid_acc.append(f1)
                
                
    torch.save(model, args.model_path)
  
    trainloss_file = '{}-train_loss-{}'.format(args.save, train_start)
    np.save(trainloss_file, train_losses)
    acc_train_file = '{}-labels_train-{}'.format(args.save, train_start)
    np.save(acc_train_file, train_acc)
    #predictions__train_file = '{}-predictions_train-{}'.format(args.save, train_start)
    #np.save(predictions__train_file, all_predictions_train)
      

    # safe valid data
    validloss_file = '{}-valid_loss-{}'.format(args.save, train_start)
    np.save(validloss_file, valid_losses)
    acc_valid_file = '{}-acc_valid-{}'.format(args.save, train_start)
    np.save(acc_valid_file, valid_acc)
    #predictions__valid_file = '{}-predictions_valid-{}'.format(args.save, train_start)
    #np.save(predictions__valid_file, all_predictions_valid)
            
        
    test_losses = []
    all_labels_test = []
    all_predictions_test = []

    #train_start = time.strftime("%Y%m%d-%H%M")
  
    for epoch in range(args.test_epochs):
      
        labels, predictions, test_loss = Valid(model, train_queue, valid_queue, optimizer, criterion, device, args.test_num_steps, args.report_freq)
          
        test_losses.append(test_loss)
        all_labels_test.append(labels)
        all_predictions_test.append(predictions)
      

    testloss_file = '{}-test_loss-{}'.format(args.save, train_start)
    np.save(testloss_file, test_losses)
    labels_test_file = '{}-labels_test-{}'.format(args.save, train_start)
    np.save(labels_test_file, all_labels_test)
    predictions_test_file = '{}-predictions_test-{}'.format(args.save, train_start)
    np.save(predictions_test_file, all_predictions_test)
    
    
