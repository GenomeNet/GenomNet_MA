#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 20:26:17 2021

@author: amadeu
"""

import argparse
import os, sys
import time
import numpy as np
import torch
import logging
import torch.nn as nn

import nas_utils.model_discCNN as one_shot_model

from nas_darts.random_Sampler import generate_random_architectures

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from nas_sampling.utils import mask2geno

from generalNAS_tools import utils

from generalNAS_tools.train_and_validate import train, infer

from generalNAS_tools.utils import overall_acc, overall_f1


parser = argparse.ArgumentParser(description='DARTS for genomic Data')

parser.add_argument('--task', type=str, default='TF_bindings', help='defines the task: next_character_prediction (not fully implemented!) or TF_bindings (default)')

parser.add_argument('--seed', type=int, default=3, help='random seed')

parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--cnn_lr', type=float, default=0.025, help='learning rate for CNN part')
parser.add_argument('--cnn_weight_decay', type=float, default=3e-4, help='weight decay for CNN part')
parser.add_argument('--rhn_lr', type=float, default=2, help='learning rate for RHN part')
parser.add_argument('--rhn_weight_decay', type=float, default=5e-7, help='weight decay for RHN part')

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
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')


parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='batch size')


parser.add_argument('--num_samples', type=int, default=3, help='number of random sampled architectures')


parser.add_argument('--train_directory', type=str, default='data/deepsea_train/train.mat', help='file (TF_bindings) or directory (next_character_prediction) of training data')
parser.add_argument('--valid_directory', type=str, default='data/deepsea_train/valid.mat', help='file (TF_bindings) or directory (next_character_prediction) of validation data')
parser.add_argument('--test_directory', type=str, default='data/deepsea_train/test.mat', help='file (TF_bindings) or directory (next_character_prediction) of test data')

parser.add_argument('--seq_size', type=int, default=1000, help='input sequence size')

parser.add_argument('--next_character_predict_character', dest='next_character_prediction', action='store_true', help='only for --task=next_character_prediction: predict single character')
parser.add_argument('--next_character_predict_sequence', dest='next_character_prediction', action='store_false', help='only for --task=next_character_prediction: predict entire sequence, shifted by one character, using causal convolutions')
parser.set_defaults(next_character_prediction=True)

parser.add_argument('--num_files', type=int, default=3, help='number of files for training data (for --task=next_character_prediction)')


parser.add_argument('--report_freq', type=int, default=1, metavar='N', help='report interval')

parser.add_argument('--save', type=str,  default='search', help='file name postfix to save the labels and predicitons')
parser.add_argument('--save_dir', type=str,  default= 'test_search', help='path to save the labels and predicitons')

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
           
        
    random_architectures = generate_random_architectures(generate_num=args.num_samples) # generate <num_samples> random adj. matrices
    
    cnn_masks = []
    rnn_masks = []
    for cnn_sub in random_architectures:
        cnn_masks.append(cnn_sub[0])
    for rnn_sub in random_architectures:
        rnn_masks.append(rnn_sub[1])
            
    random_search_results = []
    
    count = 0

    # store metrics of each configuration here
    acc_valid_all = []
    
    for mask in random_architectures: # iterate over all architectures
        
        configuration_start = time.time()

        count += 1
        
        genotype = mask2geno(mask)
        
        logging.info('| num_architecture {:3d}'.format(count))
           
        logging.info(genotype)   

        multiplier, stem_multiplier = 4, 3
        
        random_model = one_shot_model.RNNModel(args.seq_size, args.dropouth, args.dropoutx,
                            args.init_channels, num_classes, args.layers, args.steps, multiplier, stem_multiplier,
                            False, 0.2, genotype=genotype, task=args.task).to(device)
        
        conv = []
        rhn = []
        for name, param in random_model.named_parameters():
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
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
        
        clip_params = [args.conv_clip, args.rhn_clip, args.clip]
        
        # metrics for a configuration (epoch wise)
        valid_acc = []

        for epoch in range(args.epochs):
            logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
    
            lr = scheduler.get_last_lr()[0]
        
            labels, predictions, train_loss = train(train_queue, valid_queue, random_model, rhn, conv, criterion, optimizer, None, None, args.unrolled, lr, epoch, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, train_arch=False, pdarts=False, task=args.task)
            
            scheduler.step()
    
            labels = np.concatenate(labels)
            predictions = np.concatenate(predictions)

            if args.task == 'next_character_prediction':
                acc = overall_acc(labels, predictions, args.task)
                logging.info('| epoch {:3d} | train acc {:5.2f}'.format(epoch, acc))
            else:
                f1 = overall_f1(labels, predictions, args.task)
                logging.info('| epoch {:3d} | train f1-score {:5.2f}'.format(epoch, f1))

        labels, predictions, valid_loss = infer(valid_queue, random_model, criterion, args.batch_size, args.num_steps, args.report_freq, task=args.task)

        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
  
        if args.task == 'next_character_prediction':
            acc = overall_acc(labels, predictions, args.task)
            logging.info('| epoch {:3d} | valid acc {:5.2f}'.format(epoch, acc))
            valid_acc.append(acc)

        else:
            acc = overall_f1(labels, predictions, args.task)
            logging.info('| epoch {:3d} | valid f1-score {:5.2f}'.format(epoch, acc))
            valid_acc.append(acc)
                    
        random_search_results.append((genotype, acc)) # wenn wir z.B. 20 random samples haben, dann hätten wir liste mit 20 elementen
          
        acc_valid_all.append(valid_acc)
       
         
    def acc_position(list):
        return list[1]
    
    random_search_results.sort(reverse=True, key=acc_position)

    # final/best architecture
    genotype = random_search_results[0][0]

    genotype_file = 'random_geno-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, genotype_file), genotype)

    acc_valid_file = 'acc_valid-{}'.format(args.save)
    np.save(os.path.join(args.save_dir, acc_valid_file), acc_valid_all)
    
if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    duration_m = duration/60
    duration_h = duration_m/60
    duration_d = duration_h/24
    logging.info('Total searching time: %ds', duration)
    logging.info('Total searching time: %dd', duration_d)
