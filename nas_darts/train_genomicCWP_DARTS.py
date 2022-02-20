#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 23:09:23 2021

@author: amadeu
"""

import os
import sys
import time
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import copy

import nas_utils.model_search as one_shot_model

from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from generalNAS_tools.train_and_validate import train, infer

from generalNAS_tools import utils

from darts_tools.final_stage_run import final_stage_genotype
from darts_tools.auxiliary_functions import parse_network, check_sk_number, delete_min_sk_prob, keep_1_on, keep_2_branches
from darts_tools.discard_operations import discard_cnn_ops, discard_rhn_ops

from generalNAS_tools.utils import overall_acc, overall_f1


parser = argparse.ArgumentParser("CWP-DARTS for genomic data")

parser.add_argument('--task', type=str, default='TF_bindings', help='defines the task: next_character_prediction (not fully implemented!) or TF_bindings (default)')

parser.add_argument('--seed', type=int, default=2, help='random seed')

parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--cnn_lr', type=float, default=0.025, help='learning rate for CNN part')
parser.add_argument('--cnn_weight_decay', type=float, default=3e-4, help='weight decay for CNN part')
parser.add_argument('--rhn_lr', type=float, default=2, help='learning rate for RHN part')
parser.add_argument('--rhn_weight_decay', type=float, default=5e-7, help='weight decay for RHN part')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')

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
parser.add_argument('--epochs', type=int, default=[25, 15, 15], help='num of training epochs') # [10, 9, 8, 7, 6, 5, 5]

parser.add_argument('--steps', type=int, default=4, help='total number of Nodes')

parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='batch size')


parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--dropout_rate', action='append', default=[0.1, 0.1, 0.1], help='dropout rate of skip connect')
parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
parser.add_argument('--add_layers', action='append', default=['0'], help='add layers')


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
    
    
    num_to_keep = [6, 3, 1] # num_to_keep = [5, 3, 1]
    num_to_drop = [3, 3, 2] 
    
    num_to_keep_rnn = [3, 2, 1]
    num_to_drop_rnn = [2, 1, 1]
    disc_rhn_ops = np.nonzero(num_to_drop_rnn)

    # how many channels are added
    if len(args.add_width) == 3:
        add_width = args.add_width
    else:
        add_width = [0, 0, 0]
        
    # how many layers are added
    if len(args.add_layers) == 3:
        add_layers = args.add_layers
    else:
        add_layers = [0, 0, 0]
        
    if len(args.dropout_rate) ==3:
        drop_rate = args.dropout_rate
    else:
        drop_rate = [0.0, 0.0, 0.0]
        
    # num of epochs without alpha weight updates    
    eps_no_archs = [15, 3, 3]
    

    train_losses = []
    valid_losses = []
    train_acc = []
    valid_acc = []
    time_per_epoch = []

    clip_params = [args.conv_clip, args.rhn_clip, args.clip]
    
    # iterate over stages
    for sp in range(len(num_to_keep)): 
    
        if sp == 0:
       
           
           k_cnn = sum(1 for i in range(args.steps) for n in range(2+i))
          
           num_ops_cnn = len(switches_normal_cnn[0])
           
           alphas_normal = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_cnn, num_ops_cnn))) # vorher: k_cnn, num_ops_cnn
           alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_cnn, num_ops_cnn))) # vorher: k_cnn, num_ops_cnn
           
           k_rhn = sum(i for i in range(1, rnn_steps+1))             
           num_ops_rhn = len(switches_rnn[0])
           
           alphas_rnn = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k_rhn, num_ops_rhn)))
           
           multiplier, stem_multiplier = 4,3

           model = one_shot_model.RNNModelSearch(args.seq_size, args.dropouth, args.dropoutx,
                          args.init_channels, num_classes, args.layers, args.steps, multiplier, stem_multiplier,  
                          True, 0.2, None, args.task, 
                          switches_normal_cnn, switches_reduce_cnn, switches_rnn, float(drop_rate[sp]), alphas_normal, alphas_reduce, alphas_rnn).to(device)
                              
        if sp > 0:
            
            old_dict = model.state_dict()
          
            new_reduce_arch = nn.Parameter(new_reduce_arch)
            new_normal_arch = nn.Parameter(new_normal_arch)
            new_rnn_arch = nn.Parameter(new_arch_rnn)
            
            model = one_shot_model.RNNModelSearch(args.seq_size, args.dropouth, args.dropoutx,
                          args.init_channels, num_classes, args.layers, args.steps, multiplier, stem_multiplier,  
                          True, 0.2, None, args.task, 
                          switches_normal_cnn, switches_reduce_cnn, switches_rnn, float(drop_rate[sp]), new_normal_arch, new_reduce_arch, new_rnn_arch).to(device)
                              

            new_dict = model.state_dict()
           
            trained_weights = {k: v for k, v in old_dict.items() if k in new_dict}
            # need to be the old weights
           
            trained_weights["alphas_normal"] = new_normal_arch 
            trained_weights["alphas_reduce"] = new_reduce_arch
            trained_weights["weights"] = new_rnn_arch
            trained_weights["rnns.0.weights"] = new_rnn_arch
            
            new_dict.update(trained_weights)

            model.load_state_dict(new_dict) 
            
    
        model = model.to(device)
        
    
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        
        conv = []
        rhn = []
        for name, param in model.named_parameters(): # 1280 elements (because not listed each on its own; this is like batchnorma)
           if 'rnns' in name:
               rhn.append(param)
           else:
               conv.append(param)
        
        optimizer = torch.optim.SGD([{'params': conv}, {'params':rhn}], lr=args.cnn_lr, weight_decay=args.cnn_weight_decay)
        optimizer.param_groups[0]['lr'] = args.cnn_lr
        optimizer.param_groups[0]['weight_decay'] = args.cnn_weight_decay
        optimizer.param_groups[0]['momentum'] = args.momentum
        optimizer.param_groups[1]['lr'] = args.rhn_lr
        optimizer.param_groups[1]['weight_decay'] = args.rhn_weight_decay
        
    
        # optimizer for alpha updates
        optimizer_a = torch.optim.Adam(model.arch_parameters(),
                    lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs[sp]), eta_min=args.learning_rate_min)
        
        sm_dim = -1
        epochs = args.epochs[sp]
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2
        
        for epoch in range(epochs):
            train_start = time.strftime("%Y%m%d-%H%M")

            
            lr = scheduler.get_last_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()
            if epoch < eps_no_arch: 
                model.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs 
                model.update_p()       
                labels, predictions, train_loss = train(train_queue, valid_queue, model, rhn, conv, criterion, optimizer, optimizer_a, None, args.unrolled, lr, epoch, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, train_arch=False, pdarts=True, task=args.task)

            else:
                model.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor) 
                model.update_p()  
                labels, predictions, train_loss = train(train_queue, valid_queue, model, rhn, conv, criterion, optimizer, optimizer_a, None, args.unrolled, lr, epoch, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, train_arch=True, pdarts=True, task=args.task)
            
            
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
            
            
            if args.validation == True:
                if epoch % args.report_validation == 0:
                    labels, predictions, valid_loss = infer(valid_queue, model, criterion, args.batch_size, args.num_steps, args.report_freq, task=args.task)
                    
                    labels = np.concatenate(labels)
                    predictions = np.concatenate(predictions)
                    
                    valid_losses.append(valid_loss)
                   
                    if args.task == 'next_character_prediction':
                        acc = overall_acc(labels, predictions, args.task)
                        logging.info('| epoch {:3d} | valid acc {:5.2f}'.format(epoch, acc))
                        valid_acc.append(acc)

                    else:
                        f1 = overall_f1(labels, predictions, args.task)
                        logging.info('| epoch {:3d} | valid f1-score {:5.2f}'.format(epoch, f1))
                        valid_acc.append(f1)

        
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        print('------Dropping %d paths------' % num_to_drop[sp])
        # Save switches info for s-c refinement. 
        if sp == len(num_to_keep) - 1: 
            switches_normal_2 = copy.deepcopy(switches_normal_cnn)
            switches_reduce_2 = copy.deepcopy(switches_reduce_cnn)
            switches_rnn2 = copy.deepcopy(switches_rnn)
            
               
        new_normal_arch, new_reduce_arch, switches_normal_cnn, switches_reduce_cnn = discard_cnn_ops(model, switches_normal_cnn, switches_reduce_cnn, num_to_keep, num_to_drop, sp, new_alpha_values=True)
       
        if sp in disc_rhn_ops[0]:
            new_arch_rnn, switches_rnn = discard_rhn_ops(model, switches_rnn, num_to_keep_rnn, num_to_drop_rnn, sp, new_alpha_values=True)
       
        if sp == len(num_to_keep) - 1: 
            
            genotype, switches_normal_cnn, switches_reduce_cnn, switches_rnn, normal_prob, reduce_prob, rnn_prob = final_stage_genotype(model, switches_normal_cnn, switches_normal_2, switches_reduce_cnn, switches_reduce_2, switches_rnn, switches_rnn2)

            logging.info(genotype)
            logging.info('Restricting skipconnect...')
            # regularization of skip connections
            for sks in range(0, 9): 
                # sks=8
                max_sk = 8 - sks                
                num_sk = check_sk_number(switches_normal_cnn) # counts number of identity/skip connections, 2
               
                if not num_sk > max_sk: # 2 > 8 for i=0 continue, 2 > 1
                    continue
                while num_sk > max_sk: # starts with 2>1
                    normal_prob = delete_min_sk_prob(switches_normal_cnn, switches_normal_2, normal_prob)
                    switches_normal_cnn = keep_1_on(switches_normal_2, normal_prob) # out of currently 4 remaining, 2 more are removed, 2 remain
                    switches_normal_cnn = keep_2_branches(switches_normal_cnn, normal_prob)
                    num_sk = check_sk_number(switches_normal_cnn)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype = parse_network(switches_normal_cnn, switches_reduce_cnn,switches_rnn)
                logging.info(genotype) 
                
            genotype_file = 'darts_cws_geno-{}'.format(args.save)
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
