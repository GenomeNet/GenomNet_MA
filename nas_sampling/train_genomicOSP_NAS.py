#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 23:17:14 2021

@author: amadeu
"""

from randomSearch_and_Hyperband_Tools.random_Sampler import generate_random_architectures
#from transform_genotype import transform_Genotype
from randomSearch_and_Hyperband_Tools.utils import mask2geno, merge, mask2switch

# von DARTS
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
import copy
import gc
import generalNAS_tools.genotypes
import randomSearch_and_Hyperband_Tools.model_search as one_shot_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from generalNAS_tools.utils import repackage_hidden, create_exp_dir, save_checkpoint

from generalNAS_tools import utils

from randomSearch_and_Hyperband_Tools.hyperbandSampler import create_cnn_supersubnet, create_rhn_supersubnet

#from randomSearch_and_Hyperband_Tools.train_and_validate import train, evaluate_architecture

from generalNAS_tools.train_and_validate_HB import train, infer

from generalNAS_tools.utils import scores_perClass, scores_Overall, pr_aucPerClass, roc_aucPerClass, overall_acc, overall_f1

# from randomSearch_and_Hyperband_Tools.hyperbandSampler import maskout_ops, create_new_supersubnet

from randomSearch_and_Hyperband_Tools.create_masks import create_cnn_masks, create_rhn_masks

import itertools
import logging

from randomSearch_and_Hyperband_Tools.hyperband_final_tools import final_stage_run


#from ..data_preprocessing import get_data
# import generalNAS_tools.data_preprocessing_new as dp


parser = argparse.ArgumentParser(description='DARTS for genomic Data')
#parser.add_argument('--data', type=str, default='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt', help='location of the data corpus')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--cnn_lr', type=float, default=0.025, help='learning rate for CNN part')
parser.add_argument('--cnn_weight_decay', type=float, default=3e-4, help='weight decay for CNN part')
parser.add_argument('--rhn_lr', type=float, default=2, help='learning rate for CNN part')
parser.add_argument('--rhn_weight_decay', type=float, default=5e-7, help='weight decay for RHN part')

parser.add_argument('--num_ops', type=int, default=4, help='number of operations which are evaluated after budget')
parser.add_argument('--pretrain_epochs', type=int, default=5, help='budget/epochs after operations are discarded')
parser.add_argument('--budget', type=int, default=1, help='budget/epochs after operations are discarded')
parser.add_argument('--num_super_subnets', type=int, default=5, help='number of supersubnets which are created after ')
parser.add_argument('--num_init_archs', type=int, default=108, help='build the initialized supermodel with a certain number of subarchitctures')


parser.add_argument('--validation', type=bool, default=True)
parser.add_argument('--report_validation', type=int, default=2, help='validation epochs') 

parser.add_argument('--num_steps', type=int, default=4, help='number of iterations per epoch')
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

parser.add_argument('--num_files', type=int, default=3, help='number of files for training data')
parser.add_argument('--seq_size', type=int, default=1000, help='sequence length')
parser.add_argument('--num_samples', type=int, default=3, help='number of random sampled architectures')
parser.add_argument('--next_character_prediction', default=True, action='store_true', help='task of model')

parser.add_argument('--one_clip', type=bool, default=True)
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--conv_clip', type=float, default=5, help='gradient clipping of convs')
parser.add_argument('--rhn_clip', type=float, default=0.25, help='gradient clipping of lstms')

parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
# parser.add_argument('--num_classes', type=int, default=4, help='num of output classes') 
parser.add_argument('--steps', type=int, default=4, help='total number of Nodes')
#parser.add_argument('--multiplier', type=int, default=4, help='multiplier')
#parser.add_argument('--stem_multiplier', type=int, default=3, help='stem multiplier')

parser.add_argument('--epochs', type=int, default=60,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
#parser.add_argument('--dropout', type=float, default=0.75,
#                    help='dropout applied to layers (0 = no dropout)')
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
parser.add_argument('--report_freq', type=int, default=1, metavar='N',
                    help='report interval')

parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--note', type=str, default='try', help='note for this run')

parser.add_argument('--save', type=str,  default='search',
                    help='name to save the labels and predicitons')
parser.add_argument('--save_dir', type=str,  default= 'test_search',
                    help='path to save the labels and predicitons')
args = parser.parse_args()


utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
#logging.getLogger().addHandler(fh)

logging = logging.getLogger(__name__)


def main():
    
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
  
    torch.manual_seed(args.seed)
      
    logging.info("args = %s", args)

    if args.task == ("next_character_prediction" or "sequence_to_sequence"):
        
        import generalNAS_tools.data_preprocessing_new as dp

        train_queue, valid_queue, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
                seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)
      
        _, test_queue, _ = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.test_directory, num_files=args.num_files,
                seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)
        
        criterion = nn.CrossEntropyLoss().to(device)
        
    if args.task == 'TF_bindings':
        
        import generalNAS_tools.data_preprocessing_TF as dp
        
        # train_queue, valid_queue, test_queue = dp.data_preprocessing(args.train_input_directory, args.valid_input_directory, args.test_input_directory, args.train_target_directory, args.valid_target_directory, args.test_target_directory, args.batch_size)
        train_queue, valid_queue, test_queue = dp.data_preprocessing(args.train_directory, args.valid_directory, args.test_directory, args.batch_size)

        criterion = nn.BCELoss().to(device)

        num_classes = 919
        
        
            
    normal_masks = create_cnn_masks(args.num_init_archs)
    reduce_masks = create_cnn_masks(args.num_init_archs)
    rnn_masks = create_rhn_masks(args.num_init_archs)

    supernet_mask = [normal_masks, reduce_masks, rnn_masks]

    #    
    
    multiplier, stem_multiplier = 4, 3

    # super_model = one_shot_model.RNNModelSearch(args.seq_size, args.dropouth, args.dropoutx,
    #                          args.init_channels, num_classes, args.layers, args.steps, multiplier, stem_multiplier,  
    #                          True, 0.2, None, args.task, 
    #                          switches_normal, switches_reduce, switches_rnn, 0.0).to(device)
    
    super_model = one_shot_model.RNNModelSearch(args.seq_size, args.dropouth, args.dropoutx,
                              args.init_channels, num_classes, args.layers, args.steps, multiplier, stem_multiplier,  
                              True, 0.2, None, args.task, supernet_mask).to(device)
        
    conv = []
    rhn = []
    for name, param in super_model.named_parameters():
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
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    
    clip_params = [args.conv_clip, args.rhn_clip, args.clip]
    
    train_losses = []
    valid_losses = []
    train_acc = []
    valid_acc = []
    time_per_epoch = []

    # pretrain a supernet S for some epochs, with all sampled edges, nodes and operations
    for epoch in range(args.pretrain_epochs): 
        # epoch=0
        # supernet.drop_path_prob = args.drop_path_prob * epoch / self.epochs
        # supernet.drop_path_prob = drop_path_prob * epoch / epochs
        lr = scheduler.get_last_lr()[0]
        
        train_start = time.strftime("%Y%m%d-%H%M")
   
        lr = scheduler.get_last_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, lr)
        epoch_start = time.time()
        
        labels, predictions, train_loss = train(train_queue, valid_queue, super_model, rhn, conv, criterion, optimizer, epoch, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, task=args.task, mask=supernet_mask)

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
            
        train_losses.append(train_loss)
        epoch_end = time.time()
        time_per_epoch.append(epoch_end)
        # train_acc, train_obj = train(train_object, super_model, criterion, optimizer, lr, epoch, rhn, conv, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip)

        scheduler.step()
        
        if args.validation == True:
                if epoch % args.report_validation == 0:
                    labels, predictions, valid_loss = infer(valid_queue, super_model, criterion, args.batch_size, args.num_steps, args.report_freq, task=args.task, mask=supernet_mask)
                    # logging.info('Valid_acc %f', valid_acc)
                    
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
        
    supernet_mask = copy.deepcopy(supernet_mask)
    
    # supernet_mask[0]
    disc_ops_normal = disc_ops_reduce = disc_ops_rhn = [1,1]
    runde=0
    
    while (len(disc_ops_normal) | len(disc_ops_reduce) | len(disc_ops_rhn) > 1):
        
        runde += 1
        print(runde)
        # create the supersubnets and evaluate them
        # cnn_test = supernet_mask[0]
        old_super_model_weights = super_model.state_dict()   
        
        hyperband_results = []
        
        mask_normal, disc_ops_normal = create_cnn_supersubnet(supernet_mask[0], args.num_ops)
        mask_reduce, disc_ops_reduce = create_cnn_supersubnet(supernet_mask[1], args.num_ops)
        mask_rhn, disc_ops_rhn = create_rhn_supersubnet(supernet_mask, args.num_ops)
        
        supernet_mask = copy.deepcopy(supernet_mask)
        print(supernet_mask)
        
        len_nor = len(disc_ops_normal)  
        len_red = len(disc_ops_reduce) 
        len_rhn = len(disc_ops_rhn)   

        iters = max(len_nor, len_red, len_rhn)
        
        # for i in range(args.num_super_subnets):
        for i in range(iters): # iters=8
           # i=2
           disc_nor = disc_ops_normal[i]
          
           disc_red = disc_ops_reduce[i]
         
           disc_rhn = disc_ops_rhn[i]
         
           disc_ops = [disc_nor, disc_red, disc_rhn]

           supersubnet_mask = maskout_ops(disc_nor, disc_red, disc_rhn, supernet_mask)

            
           # mask_normal, disc_ops_normal = create_cnn_supersubnet(supernet_mask[0], args.num_disc)
           # mask_reduce, disc_ops_reduce = create_cnn_supersubnet(supernet_mask[1], args.num_disc)
           # mask_rhn, disc_ops_rhn = create_rhn_supersubnet(supernet_mask, args.num_disc)
            
           # supersubnet_mask = [mask_normal, mask_reduce, mask_rhn]

           # validate the supersubnets using weights from pretrained supernet S
           for epoch in range(1):               
               #labels, predictions, valid_loss = infer(valid_queue, supersubnet_model, criterion, args.batch_size, 2*args.num_steps, args.report_freq, task=args.task)
               labels, predictions, valid_loss = infer(valid_queue, super_model, criterion, args.batch_size, 2*args.num_steps, args.report_freq, task=args.task, mask=supersubnet_mask)
               logging.info('| valid loss{:5.2f}'.format(valid_loss))
               
               labels = np.concatenate(labels)
               predictions = np.concatenate(predictions)
                                        
               if args.task == 'next_character_prediction':
                   acc = overall_acc(labels, predictions, args.task)
                   logging.info('| epoch {:3d} | valid acc {:5.2f}'.format(epoch, acc))
                   #valid_acc.append(acc)
               else:
                   acc = overall_f1(labels, predictions, args.task)
                   logging.info('| epoch {:3d} | valid f1-score {:5.2f}'.format(epoch, acc))
                   #valid_acc.append(acc)
               # valid_acc, valid_obj = evaluate_architecture(valid_object, supersubnet_model, criterion, optimizer, epoch) 
               hyperband_results.append([disc_ops, valid_loss]) # hätten am Ende z.B. 5 hyperband_results
           
        #try:
        def acc_position(list):
            return list[1]
        
        hyperband_results.sort(reverse=False, key=acc_position) # in descending order: best/highest acc is on top (contains the worst operations which we want to discard)
        # here, we use ascending order: best/lowest val_loss is on top (contains the worst operations which we want to discard)
        
        # discard the edges and operations from the good performing supersubnets to build new S
        # keep mask with best/highest acc 
        supernet_mask = create_new_supersubnet(hyperband_results, supernet_mask)
        
        # super_model = hyperband_results[0][1]
        
        super_model = one_shot_model.RNNModelSearch(args.seq_size, args.dropouth, args.dropoutx,
                              args.init_channels, num_classes, args.layers, args.steps, multiplier, stem_multiplier,  
                              True, 0.2, None, args.task, supernet_mask).to(device)
        
        super_model_weights = super_model.state_dict()
           
        trained_weights = {k: v for k, v in old_super_model_weights.items() if k in super_model_weights}

        super_model_weights.update(trained_weights)
           
        super_model.load_state_dict(super_model_weights) 
        
        conv = []
        rhn = []
        for name, param in super_model.named_parameters():
            #print(name)
            #if 'stem' or 'preprocess' or 'conv' or 'bn' or 'fc' in name:
            if 'rnns' in name:
                #print(name)
                rhn.append(param)
            #elif 'decoder' in name:
            else:
                #print(name)
                conv.append(param)
        
        optimizer = torch.optim.SGD([{'params':conv}, {'params':rhn}], lr=args.cnn_lr-(0.0002*runde), weight_decay=args.cnn_weight_decay)
        optimizer.param_groups[0]['lr'] = args.cnn_lr-(0.0002*runde)
        optimizer.param_groups[0]['weight_decay'] = args.cnn_weight_decay
        optimizer.param_groups[0]['momentum'] = args.momentum
        optimizer.param_groups[1]['lr'] = args.rhn_lr-(0.1*runde)
        optimizer.param_groups[1]['weight_decay'] = args.rhn_weight_decay
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.budget))
        
        clip_params = [args.conv_clip, args.rhn_clip, args.clip]
        
        # now train the remaining supernet S for few epochs (according to budget)
        for epoch in range(args.budget): 
            # epoch=0
            lr = scheduler.get_last_lr()[0]
        
            train_start = time.strftime("%Y%m%d-%H%M")
   
            lr = scheduler.get_last_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()

            # supernet.drop_path_prob = args.drop_path_prob * epoch / self.epochs
            # supernet.drop_path_prob = drop_path_prob * epoch / epochs
            labels, predictions, train_loss = train(train_queue, valid_queue, super_model, rhn, conv, criterion, optimizer, epoch, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, task=args.task, mask=supernet_mask)

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
            
            train_losses.append(train_loss)
            epoch_end = time.time()
            time_per_epoch.append(epoch_end)
                                      
            scheduler.step()
            
            if args.validation == True:
                if epoch % args.report_validation == 0:
                    labels, predictions, valid_loss = infer(valid_queue, super_model, criterion, args.batch_size, 2*args.num_steps, args.report_freq, task=args.task, mask=supernet_mask)
                    # logging.info('Valid_acc %f', valid_acc)
                   
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
                   
    
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            
        _, disc_ops_normal = create_cnn_supersubnet(supernet_mask[0], args.num_ops)
        _, disc_ops_reduce = create_cnn_supersubnet(supernet_mask[1], args.num_ops)
        _, disc_ops_rhn = create_rhn_supersubnet(supernet_mask, args.num_ops)
           

    supernet_mask, hyperband_results = final_stage_run(train_queue, valid_queue, super_model, rhn, conv, criterion, optimizer, epoch, args.num_steps, clip_params, args.report_freq, args.beta, args.one_clip, args.task, supernet_mask, args.budget, args.batch_size)



          
    # hyperband_results.sort(reverse=True, key=acc_position)
    # final/best architecture
    
    genotype = mask2geno(supernet_mask)
    
    print(genotype)
    genotype_file = 'hb_geno-{}'.format(args.save)
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

    #genotype_file = 'hb_geno-{}'.format(args.save)
    #np.save(genotype_file, genotype)
    
    #mask_file = 'hb_mask-{}'.format(args.save)
    # np.save(hyperband_file, supernet_mask, genotype)
    #np.save(mask_file, supernet_mask)
            
    #trainloss_file = 'train_loss-{}'.format(args.save, train_start)
    #np.save(trainloss_file, train_losses_all)
      
    #acctrain_file = 'acc_train-{}'.format(args.save, train_start) 
    #np.save(acctrain_file, acc_train_all)
         
    #validloss_file = 'valid_loss-{}'.format(args.save, train_start)
    #np.save(validloss_file, valid_losses_all)
      
    #accvalid_file = 'acc_valid-{}'.format(args.save, train_start) 
    #np.save(acctrain_file, acc_valid_all)
    

  

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
