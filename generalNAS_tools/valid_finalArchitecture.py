#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 08:23:43 2021

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
import time

#import genotypes_rnn
#import genotypes_cnn
from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT, Genotype

import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint 
from generalNAS_tools import utils

import generalNAS_tools.data_preprocessing_new as dp

from generalNAS_tools.train_and_validate import train, infer

from generalNAS_tools.utils import scores_perClass, scores_Overall, pr_aucPerClass, roc_aucPerClass, overall_acc, overall_f1



parser = argparse.ArgumentParser(description='Evaluate final architecture found by PDARTS')
parser.add_argument('--data', type=str, default='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt', help='location of the data corpus')
parser.add_argument('--num_steps', type=int, default=2, help='number of iterations per epoch')
parser.add_argument('--train_directory', type=str, default='/home/amadeu/Downloads/genomicData/train', help='directory of training data')
parser.add_argument('--valid_directory', type=str, default='/home/amadeu/Downloads/genomicData/validation', help='directory of validation data')
parser.add_argument('--test_directory', type=str, default='/home/amadeu/Downloads/genomicData/test', help='directory of test data')

parser.add_argument('--train_input_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/inputs_small.pkl', help='directory of train data')
parser.add_argument('--train_target_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/targets_small.pkl', help='directory of train data')
parser.add_argument('--valid_input_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/inputs_small_val.pkl', help='directory of validation data')
parser.add_argument('--valid_target_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/targets_small_val.pkl', help='directory of validation data')
parser.add_argument('--test_input_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/inputs_small_test.pkl', help='directory of test data')
parser.add_argument('--test_target_directory', type=str, default='/home/amadeu/Desktop/GenomNet_MA/data/targets_small_test.pkl', help='directory of test data')

parser.add_argument('--task', type=str, default='TF_bindings', help='defines the task')#TF_bindings


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
parser.add_argument('--steps', type=int, default=4, help='total number of Nodes')

parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')

parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
#parser.add_argument('--num_runs', type=int, default=2, metavar='N',
#                    help='number of runs')
parser.add_argument('--seq_size', type=int, default=1000,
                    help='sequence length')
parser.add_argument('--dropouth', type=float, default=0.05,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.1,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')

parser.add_argument('--save', type=str,  default='testen',
                    help='path to save the final model')
parser.add_argument('--save_dir', type=str,  default='lr8_nodrop',
                    help='path to save the labels and predicitons')

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
parser.add_argument('--genotype_file', type=str, default='/home/amadeu/Desktop/GenomNet_MA/preliminary_study_results/nodropout_rhnlr8/EXPsearch-try-20210807-072855-darts_geno.npy', help='directory of final genotype')
parser.add_argument('--search', type=bool, default=False, help='which architecture to use')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--model', type=str, default='/home/amadeu/Desktop/GenomNet_MA/nodrop_rhn8.pth', help='path to trained model')
args = parser.parse_args()


# args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    # criterion = nn.BCELoss()
  
    if (args.task == "next_character_prediction"):
        import generalNAS_tools.data_preprocessing_new as dp
      
        train_queue, valid_queue, test_queue, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, test_directory = args.test_directory, num_files=args.num_files,
                  seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)

        num_classes = 4
        criterion = nn.CrossEntropyLoss().to(device)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-6, momentum=0.9)

          
      
    if (args.task == "TF_bindings"):
      
        import generalNAS_tools.data_preprocessing_TF as dp
        
        #train_queue, valid_queue, test_queue = dp.data_preprocessing(args.train_input_directory, args.valid_input_directory, args.test_input_directory, args.train_target_directory, args.valid_target_directory, args.test_target_directory, args.batch_size)
        train_queue, valid_queue, test_queue = dp.data_preprocessing(args.train_directory, args.valid_directory, args.test_directory, args.batch_size)

        
        criterion = nn.BCELoss().to(device)
        num_classes = 919
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-6, momentum=0.9)
    

    model = torch.load(args.model)
    
    valid_losses = []
    valid_acc = []

    for epoch in range(args.epochs):
        # epoch=0
                    
        labels, predictions, valid_loss = infer(valid_queue, model, criterion, args.batch_size, args.num_steps, args.report_freq, task=args.task)
          
        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        
        if args.task == ('next_character_prediction'):
            acc = overall_acc(labels, predictions, args.task)
            logging.info('| epoch {:3d} | val acc {:5.2f}'.format(epoch, acc))
            logging.info('| epoch {:3d} | val loss {:5.2f}'.format(epoch, valid_loss))

            valid_acc.append(acc)

        else:
            f1 = overall_f1(labels, predictions, args.task)
            logging.info('| epoch {:3d} | val f1-score {:5.2f}'.format(epoch, f1))
            logging.info('| epoch {:3d} | val loss {:5.4f}'.format(epoch, valid_loss))

            valid_acc.append(f1)

      
        valid_losses.append(valid_loss)
    

    validloss_file = 'valid_loss-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, validloss_file), valid_losses)
    
    validacc_file = 'valid_acc-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, validacc_file), valid_acc)

   
    
    
if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
