#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 10:29:41 2021

@author: amadeu
"""

import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.metrics import f1_score, recall_score, precision_score, f1_score

from sklearn.metrics import classification_report
# precision, recall, f1
from sklearn.metrics import f1_score, recall_score, precision_score
# ROC PR, ROC AUC
from sklearn.metrics import roc_auc_score, auc

#from sklearn.metrics import average_precision_score, average_recall_score
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score

from sklearn.metrics import classification_report

from sklearn.preprocessing import label_binarize


import matplotlib.pyplot as plt



def scores_perClass(labels, predictions, task):
    precisions = dict()
    recalls = dict()
    f1 = dict()
    accuracies = dict()
    predictions = np.round(predictions) # to get hard labels from scores

    
    if task=='TF_bindings':
        num_classes = len(labels[1])
    
    else:
        labels = label_binarize(labels, classes=[0, 1, 2, 3])
        num_classes = labels.shape[1]
        
    for i in range(num_classes):
            # i=0
            precisions[i] = precision_score(labels[:,i], predictions[:,i])
            recalls[i] = recall_score(labels[:,i], predictions[:,i])
            f1[i] = f1_score(labels[:,i], predictions[:,i])
            accuracies[i] = accuracy_score(labels[:,i], predictions[:,i])
        
    return precisions, recalls, f1, accuracies

# prec, rec, f1, acc = scores_perClass(labels, predictions)

# print(classification_report(labels, predictions))

#from sklearn.metrics import confusion_matrix
#y_true = [2, 0, 2, 2, 0, 1]
#y_pred = [0, 0, 2, 2, 0, 2]
#matrix = confusion_matrix(y_true, y_pred)
#matrix.diagonal()/matrix.sum(axis=1)

# print(classification_report(y_true, y_pred))
# recalls[i] = recall_score(labels[:,i], predictions[:,i])



# weighted overall scores

def scores_Overall(labels, predictions, task):
    if task=='TF_bindings':
        
        predictions = np.round(predictions)
    else:
        predictions = np.argmax(predictions, axis=1)
        
    recall = recall_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions) ###########
    
    return precision, recall, f1, accuracy


def overall_acc(labels, predictions, task):
    if task=='TF_bindings':
        
        predictions = np.round(predictions)
    else:
        predictions = np.argmax(predictions, axis=1)
        
    accuracy = accuracy_score(labels, predictions) ###########
    
    return accuracy


def overall_f1(labels, predictions, task):
    if task=='TF_bindings':
        
        predictions = np.round(predictions)
    else:
        predictions = np.argmax(predictions, axis=1)
        
    f1 = f1_score(labels, predictions, average='weighted')
    
    return f1




# PR-Curve

def pr_aucPerClass(labels, predictions, task):
    
   average_precision_scores = dict()

   if task=='TF_bindings':
       num_classes = len(labels[1])
       
   else:
       labels = label_binarize(labels, classes=[0, 1, 2, 3])
       num_classes = labels.shape[1]
   for i in range(num_classes):
       # i=0
       # precisions[i], recalls[i], _ = precision_recall_curve(labels[:,i], predictions[:,i])
       average_precision_scores[i] = average_precision_score(labels[:,i], predictions[:,i])

   
   return average_precision_scores
    
    

def roc_aucPerClass(labels, predictions, task):
    roc_auc_scores = dict()

    if task=='TF_bindings':
        num_classes = len(labels[1])

        for i in range(num_classes):
            print(i)
            try:
                roc_auc_scores[i] = roc_auc_score(labels[:,i], predictions[:,i])
            except ValueError:
                pass
            
    else:
        labels = label_binarize(labels, classes=[0, 1, 2, 3])
        num_classes = labels.shape[1]
        fpr = dict()
        tpr = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
            roc_auc_scores[i] = auc(fpr[i], tpr[i])
        
    return roc_auc_scores



class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


# output, target, topk = logits, label,  (1,2)
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    #correct_k = correct[:k].view(-1).float().sum(0)
    correct_k = correct[:k].reshape(k*batch_size).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.FloatTensor(x.size(0), 1, 1).bernoulli_(keep_prob))#Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask.to(device))
  return x.to(device)


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
      
      
#def save_checkpoint(model, optimizer, epoch, path, finetune=False):
#    if finetune:
#        torch.save(model, os.path.join(path, 'finetune_model.pt'))
#        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
#    else:
#        torch.save(model, os.path.join(path, 'model.pt'))
#        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
#    torch.save({'epoch': epoch+1}, os.path.join(path, 'misc.pt'))

        


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


def mask2d(B, D, keep_prob, cuda=True): 

    
    m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
    
    m = Variable(m, requires_grad=False)
    if cuda:
        m = m.to(device)
    return m 
