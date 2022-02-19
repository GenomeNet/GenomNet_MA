#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:33:30 2021

@author: amadeu
"""

import os
import sys
import time
import glob
import numpy as np
import torch
# import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import gc

from generalNAS_tools import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import csv
import gc
#import DeepVirFinder_model as dvf
import generalNAS_tools.data_preprocessing_new as dp
import torch
#from DeepVirFinder_model import model, device, criterion, optimizer, num_motifs, num_classes
#from data_preprocessing import train_loader, valid_loader, batch_size, num_batches_valid, num_batches_train, valid_y

import torch.nn as nn
import numpy as np 
from sklearn.metrics import classification_report
# precision, recall, f1
from sklearn.metrics import f1_score, recall_score, precision_score
# ROC PR, ROC AUC
from sklearn.metrics import roc_auc_score, auc

from sklearn.metrics import precision_recall_curve
from generalNAS_tools.utils import scores_perClass, scores_Overall, pr_aucPerClass, roc_aucPerClass, overall_acc, overall_f1

import matplotlib.pyplot as plt


# from pytorchtools import EarlyStopping

parser = argparse.ArgumentParser("train_baseline")
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--seq_size', type=int, default=1000, help='input sequence size') # 200 or 1000
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate') # default RMSprob
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs') # min 60 but until convergence
# parser.add_argument('--test_epochs', type=int, default=1, help='num of testing epochs')

parser.add_argument('--task', type=str, default='TF_bindings', help='defines the task: next_character_prediction (not fully implemented!) or TF_bindings (default)')
parser.add_argument('--num_steps', type=int, default=3000, help='number of iterations per epoch')

parser.add_argument('--train_directory', type=str, default='data/deepsea_train/train.mat', help='file (TF_bindings) or directory (next_character_prediction) of training data')
parser.add_argument('--valid_directory', type=str, default='data/deepsea_train/valid.mat', help='file (TF_bindings) or directory (next_character_prediction) of validation data')
parser.add_argument('--test_directory', type=str, default='data/deepsea_train/test.mat', help='file (TF_bindings) or directory (next_character_prediction) of test data')

parser.add_argument('--num_files', type=int, default=3, help='number of files for training data')
parser.add_argument('--next_character_predict_character', dest='next_character_prediction', action='store_true', help='only for --task=next_character_prediction: predict single character')
parser.add_argument('--next_character_predict_sequence', dest='next_character_prediction', action='store_false', help='only for --task=next_character_prediction: predict entire sequence, shifted by one character, using causal convolutions')
parser.set_defaults(next_character_prediction=True)

# parser.add_argument('--num_motifs', type=int, default=100, help='number of channels') # 320
parser.add_argument('--model', type=str, required=True, help='Model to use; one of: DanQ, DeepSEA (note case sensitive), NCNet_RR, NCNet_bRR')
parser.add_argument('--save', type=str, required=True,
                    help='file name postfix to save the labels and predicitons')
parser.add_argument('--save_dir', type=str, required=True,
                    help='path to save the labels and predicitons')
parser.add_argument('--model_path', type=str, required=True,
                    help='path to save the trained model')

parser.add_argument('--report_validation', type=int, default=1, help='validation report period; default 1 (every epoch)')
parser.add_argument('--improv', type=float, default=0, help='minimum change to define an improvement for early stopping; default 0')
parser.add_argument('--patience', type=int, default=0, help='how many epochs with no improvement after which training will be stopped; default 0: disabled.')
args = parser.parse_args()


utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging = logging.getLogger(__name__)


def main():
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  if (args.task == "next_character_prediction"):
      
      import generalNAS_tools.data_preprocessing_new as dp
      
      train_queue, valid_queue, num_classes = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.valid_directory, num_files=args.num_files,
                seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)
      
      _, test_queue, _ = dp.data_preprocessing(train_directory = args.train_directory, valid_directory = args.test_directory, num_files=args.num_files,
                seq_size = args.seq_size, batch_size=args.batch_size, next_character=args.next_character_prediction)
      criterion = nn.CrossEntropyLoss().to(device)
      num_classes = 4


      if (args.model == "DanQ"):
          import baseline_models.models.DanQ_model as model

          model = model.NN_class(num_classes, args.batch_size, args.seq_size, args.task).to(device)
          
          optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
          
      if (args.model == "DeepSEA"):
          
          import baseline_models.models.DeepSea as model
          from baseline_models.tain_and_val_deepsea import Train, Valid

          model = model.NN_class(num_classes, args.batch_size, args.seq_size, args.task).to(device)

          optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-07, momentum=0.9)
        
      
      
  if (args.task == "TF_bindings"):
      
      import generalNAS_tools.data_preprocessing_TF as dp
        
      train_queue, valid_queue, test_queue = dp.data_preprocessing(args.train_directory, args.valid_directory, args.test_directory, args.batch_size)

      num_classes = 919
      
      criterion = nn.BCELoss().to(device)

      if (args.model == "DanQ"):
          
          import baseline_models.models.DanQ_model as model
          model = model.NN_class(num_classes, args.batch_size, args.seq_size, args.task).to(device)

          from baseline_models.tain_and_val_baseline import Train, Valid, Test


          #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-6, momentum=0.9)

          optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.9) 
          
      if (args.model == "DeepSEA"): 
          
          import baseline_models.models.DeepSea as model
          from baseline_models.tain_and_val_deepsea import Train, Valid, Test
          model = model.NN_class(num_classes, args.batch_size, args.seq_size, args.task).to(device)
          optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-07, momentum=0.9)
          
      if (args.model == 'NCNet_RR'): 
                   
          import baseline_models.models.NCNet_RR_model as model
          from baseline_models.tain_and_val_baseline import Train, Valid, Test
          from baseline_models.models.NCNet_RR_model import ResidualBlock
        
          # define hyperparameters
          net_args = {
            "res_block": ResidualBlock,
            "seq_size": args.seq_size,
            "num_classes": num_classes,
            "batch_size": args.batch_size,
            "task": args.task
            }
          
          model = model.NCNet_RR(**net_args).to(device)
                    
          optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.9)
          
          
      if (args.model == 'NCNet_bRR'): 
                   
          import baseline_models.models.NCNet_bRR_model as model
          from baseline_models.tain_and_val_baseline import Train, Valid, Test
          from baseline_models.models.NCNet_bRR_model import ProjectionBlock, IdentityBlock
        
          # define hyperparameters
          net_args = {
            "id_block": IdentityBlock,
            "pr_block": ProjectionBlock,
            "seq_size": args.seq_size,
            "num_classes": num_classes,
            "batch_size": args.batch_size,
            "task": args.task
            }
          
          model = model.NCNet_bRR(**net_args).to(device)
                    
          optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.9)

 
  train_losses = []
  valid_losses = []
  train_acc = []
  #all_predictions_train = []
  valid_acc = []
  #all_predictions_valid = []
  time_per_epoch = []
  cnt=0
  
  best_loss = float('inf')
  
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  
  pytorch_total_params = sum(p.numel() for p in model.parameters()) 

  
  # pytorch_total_params = sum(p.numel() for p in model.parameters()) # 46e6 with danQ; 57e6 with NCNet_RR; 38e6 with DARTS oneshot model; 22e6 with final archs
  
  # mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
  # mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
  # mem = mem_params + mem_bufs 
  
  # params = []
  # names = []
  # numels = []
  # for name, p in model.named_parameters():
  #    params.append(p)
  #    names.append(name)
  #    numels.append(p.numel()) # 10th element is highest und fc after lstm, because it has shape [925,48000] and p.numel for this is 925*48000(since 640*75=48.000)= 44.400.000; these are summed up (with train_finalArchs these are 22e6, since 925*21504(weil 512*42=21504))=19.891.200
      


  for epoch in range(args.epochs):
      
      train_start = time.strftime("%Y%m%d-%H%M")
      epoch_start = time.time()

      labels, predictions, train_loss = Train(model, train_queue, optimizer, criterion, device, args.num_steps, args.task)

      labels = np.concatenate(labels)
      predictions = np.concatenate(predictions)
      
      if args.task == 'next_character_prediction':
          acc = overall_acc(labels, predictions, args.task)
          
          logging.info('| epoch {:3d} | train acc {:5.2f}'.format(epoch, acc))
          train_acc.append(acc)


      else:
          f1 = overall_f1(labels, predictions, args.task)
          logging.info('| epoch {:3d} | train f1-score {:5.4f}'.format(epoch, f1))
          train_acc.append(f1)
          

      #all_predictions_train.append(predictions)
      train_losses.append(train_loss)
      epoch_end = time.time()
      time_per_epoch.append(epoch_end)
      epoch_end = time.time()
      epoch_duration = epoch_end - epoch_start
      logging.info('Epoch time: %ds', epoch_duration)

      # acc_train.append(acc_train_epoch)
      # np.save('train_loss', train_losses)
      # np.save('acc_train', acc_train)
      
      if epoch % args.report_validation == 0:
          
          labels, predictions, valid_loss = Valid(model, valid_queue, optimizer, criterion, device, args.num_steps, args.task)
          
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
              
          
          if valid_loss < best_loss:
              best_loss = valid_loss
              cnt = 0
          else:
              cnt += 1
              print(cnt)
             
          valid_losses.append(valid_loss)
           
          # Early stopping
          #if epoch > 0:
          #    if (valid_loss+args.improv) < valid_losses[epoch-1]:
          #        cnt = 0
          #    else:
          #        cnt += 1
          #    print(cnt)
                  
          if args.patience and cnt == args.patience:
              
              break
            
              # valid_losses.append(valid_loss)
              # torch.save(model, args.model_path)
            
              # trainloss_file = 'train_loss-{}'.format(args.save)
              # np.save(os.path.join(args.save_dir, trainloss_file), train_losses)
         
              # acc_train_file = 'acc_train-{}'.format(args.save)
              # np.save(os.path.join(args.save_dir, acc_train_file), train_acc)

              # predictions__train_file = '{}-predictions_train-{}'.format(args.save, train_start)
              # np.save(predictions__train_file, all_predictions_train)

              # time_file = 'time-{}'.format(args.save)
              # np.save(os.path.join(args.save_dir, time_file), time_per_epoch)

              # safe valid data
              # validloss_file = 'valid_loss-{}'.format(args.save)
              # np.save(os.path.join(args.save_dir, validloss_file), valid_losses)

              # acc_valid_file = 'acc_valid-{}'.format(args.save)
              # np.save(os.path.join(args.save_dir, acc_valid_file), valid_acc)
                    
                          
         
  torch.save(model, args.model_path)

  trainloss_file = 'train_loss-{}'.format(args.save)
  np.save(os.path.join(args.save_dir, trainloss_file), train_losses)
 
  acc_train_file = 'acc_train-{}'.format(args.save)
  np.save(os.path.join(args.save_dir, acc_train_file), train_acc)
  
  # predictions__train_file = '{}-predictions_train-{}'.format(args.save, train_start)
  # np.save(predictions__train_file, all_predictions_train)

  time_file = 'time-{}'.format(args.save)
  np.save(os.path.join(args.save_dir, time_file), time_per_epoch)
  
  # safe valid data
  validloss_file = 'valid_loss-{}'.format(args.save)
  np.save(os.path.join(args.save_dir, validloss_file), valid_losses)

  acc_valid_file = 'acc_valid-{}'.format(args.save)
  np.save(os.path.join(args.save_dir, acc_valid_file), valid_acc)

  
  #### test ### 
  test_losses = []
  all_labels_test = []
  all_predictions_test = []

  train_start = time.strftime("%Y%m%d-%H%M")
  
  labels, predictions, test_loss = Test(model, test_queue, optimizer, criterion, device, args.num_steps, args.task)

  
  #for epoch in range(args.test_epochs): # 1 epoch has 3000 iterations time 32 batchsize is 96000 samples
  
  #objs = utils.AvgrageMeter()
  #top1 = utils.AvgrageMeter()
  #top5 = utils.AvgrageMeter()
    
  #total_loss = 0
  #start_time = time.time()
    
  #model.eval()

  #labels = []
  #predictions = []
  #scores = nn.Softmax()

  #with torch.no_grad():
        
  #    for idx, (inputs, targets) in enumerate(test_queue):
        
  #        input, label = inputs, targets
               
  #        input = input.float().to(device)#.cuda()
            
          #label = torch.max(label, 1)[1]
  #        label = label.to(device)#.cuda(non_blocking=True)
  #        batch_size = input.size(0)
        
          #if args.task == "TF_bindings":
  #        logits = model(input.float()) #, (state_h, state_c))

  #        loss = criterion(logits, label)
          #else:
              # label = torch.max(label, 1)[1]
          #logits = model(input.float()) #, (state_h, state_c))
            
  #        labels.append(label.detach().cpu().numpy())
  #        if args.task == "next_character_prediction":
  #            predictions.append(scores(logits).detach().cpu().numpy())
  #        else:
  #            predictions.append(logits.detach().cpu().numpy())
                
  #        objs.update(loss.data, batch_size)
  #  
          
  labels = np.concatenate(labels)
  predictions = np.concatenate(predictions)
      
  test_losses.append(test_loss)
  all_labels_test.append(labels)
  all_predictions_test.append(predictions)
      

  testloss_file = 'test_loss-{}'.format(args.save)
  np.save(os.path.join(args.save_dir, testloss_file), test_losses)

  labels_test_file = 'labels_test-{}'.format(args.save)
  np.save(os.path.join(args.save_dir, labels_test_file), all_labels_test)

  predictions_test_file = 'predictions_test-{}'.format(args.save)
  np.save(os.path.join(args.save_dir, predictions_test_file), all_predictions_test)
  
  # 455024/100
  
  # 4550*100
  
      

if __name__ == '__main__':
  main() 
