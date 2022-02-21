#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:21:29 2021

@author: amadeu
"""

import numpy as np
import torch
import opendomain_utils.training_utils as utils
import logging
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import random

from BONAS_search_space.super_model import geno2mask, merge

import nas_bonas.model_search as one_shot_model


from opendomain_utils.ioutils import copy_log_dir
from opendomain_utils.genotypes import Genotype




from itertools import cycle, islice

from opendomain_utils.bn_utils import set_running_statistics
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import gc

from generalNAS_tools.utils import scores_perClass, scores_Overall, pr_aucPerClass, roc_aucPerClass, overall_acc, overall_f1, count_parameters_in_MB, AvgrageMeter
from generalNAS_tools.train_and_validate import train, infer


class Trainer:
    def __init__(self,
                 train_supernet_epochs=5,
                 data_path='data',
                 super_batch_size=2,
                 sub_batch_size=4,
                 
                 cnn_lr=0.025,
                 cnn_weight_decay=3e-4,
                 rhn_lr=2,
                 rhn_weight_decay=5e-7,
                 momentum=0.9,
                 
                 report_freq=2,
                 num_steps=2,
                 epochs=50,
                 init_channels=8,
                 layers=3,
                 drop_path_prob=0.2,
                 seed=0,
                 parallel = False,
                 mode='uniform',
                 train_directory = '/home/amadeu/Downloads/genomicData/train',
                 valid_directory = '/home/amadeu/Downloads/genomicData/validation',
                 test_directory = '/home/amadeu/Downloads/genomicData/validation',
                 
                 
                 train_input_directory = '/home/amadeu/Desktop/GenomNet_MA/data/inputs_small.pkl',
                 train_target_directory = '/home/amadeu/Desktop/GenomNet_MA/data/targets_small.pkl',
                 valid_input_directory = '/home/amadeu/Desktop/GenomNet_MA/data/inputs_small_val.pkl',
                 valid_target_directory = '/home/amadeu/Desktop/GenomNet_MA/data/targets_small_val.pkl',
                 test_input_directory = '/home/amadeu/Desktop/GenomNet_MA/data/inputs_small_test.pkl',
                 test_target_directory = '/home/amadeu/Desktop/GenomNet_MA/data/targets_small_test.pkl',
               
                 task = 'TF_bindings',

                 num_files = 3,
                 seq_len = 120,
                 dropouth = 0.25,
                 dropoutx = 0.75,
                 next_character_prediction=True,
                 one_clip=True,
                 clip=5,
                 conv_clip=5,
                 rhn_clip=0.25,
                 steps = 4,
                 multiplier = 4, 
                 stem_multiplier = 3,
                 beta =1e-3
                 ):
        
        self.parallel = parallel
        self.train_supernet_epochs = train_supernet_epochs
        self.data_path = data_path
        self.super_batch_size = super_batch_size
        self.sub_batch_size = sub_batch_size
        
        self.cnn_lr = cnn_lr
        self.cnn_weight_decay = cnn_weight_decay
        self.rhn_lr = rhn_lr
        self.rhn_weight_decay = rhn_weight_decay
        self.momentum = momentum
        
        self.mode = mode
        
        self.report_freq = report_freq
        self.epochs = epochs
        self.init_channels = init_channels
        self.layers = layers
        self.drop_path_prob = drop_path_prob
        self.seed = seed
        #self.criterion = nn.CrossEntropyLoss().to(device)
        self.train_directory = train_directory
        self.valid_directory = valid_directory
        self.test_directory = test_directory
        
        self.train_input_directory = train_input_directory
        self.train_target_directory = train_target_directory
        self.valid_input_directory = valid_input_directory
        self.valid_target_directory = valid_target_directory
        self.test_input_directory = test_input_directory
        self.test_target_directory = test_target_directory
        self.task = task

        self.num_files = num_files
        self.seq_len = seq_len
        self.num_steps = num_steps
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        self.steps = steps
        self.multiplier = multiplier
        self.stem_multiplier = stem_multiplier
        self.next_character_prediction = next_character_prediction
        self.one_clip = one_clip
        self.clip = clip
        self.conv_clip=conv_clip
        self.rhn_clip=rhn_clip
        self.beta = beta

        self.build_dataloader()

        self.train_loader_super, self.train_loader_sub, self.valid_loader, self.num_classes = self.build_dataloader()
        # train_loader_super, train_loader_sub, valid_loader = build_dataloader()



    
    
    # only difference between train_loader_sub and train_loader_super is different batch_size, and one is for the supermodel and the
    # other for the subnets
    
    
    def build_dataloader(self):
                
        if self.task == ("next_character_prediction" or "sequence_to_sequence"):
        
            import generalNAS_tools.data_preprocessing_new as dp

            train_loader_super, valid_queue, num_classes = dp.data_preprocessing(train_directory = self.train_directory, valid_directory = self.valid_directory, num_files=self.num_files,
                                                    seq_size = self.seq_len, batch_size=self.super_batch_size, next_character=self.next_character_prediction)
      
            train_loader_sub, valid_loader, num_classes = dp.data_preprocessing(train_directory = self.train_directory, valid_directory = self.valid_directory, num_files=self.num_files,
                                                    seq_size = self.seq_len, batch_size=self.sub_batch_size, next_character=self.next_character_prediction)
            #_, test_queue, _ = dp.data_preprocessing(train_directory = self.train_directory, valid_directory = self.test_directory, num_files=self.num_files,
            #                                         seq_size = self.seq_len, batch_size=self.batch_size, next_character=self.next_character_prediction)
            
            self.criterion = nn.CrossEntropyLoss().to(device)
            self.num_classes = 4
        
        if self.task == 'TF_bindings':
        
            import generalNAS_tools.data_preprocessing_TF as dp
            
            self.num_classes = 919

        
            #train_loader_super, valid_queue, test_queue = dp.data_preprocessing(self.train_input_directory, self.valid_input_directory, self.test_input_directory, self.train_target_directory, self.valid_target_directory, self.test_target_directory, self.super_batch_size)
            train_loader_super, valid_queue, test_queue = dp.data_preprocessing(self.train_directory, self.valid_directory, self.test_directory, self.super_batch_size)

            # train_loader_sub, valid_loader, test_queue = dp.data_preprocessing(self.train_input_directory, self.valid_input_directory, self.test_input_directory, self.train_target_directory, self.valid_target_directory, self.test_target_directory, self.sub_batch_size)
            train_loader_sub, valid_loader, test_queue = dp.data_preprocessing(self.train_directory, self.valid_directory, self.test_directory, self.sub_batch_size)
            
            self.criterion = nn.BCELoss().to(device)
        
            # train_loader_sub, valid_loader, num_classes = dp.data_preprocessing(self.train_directory, self.valid_directory, self.num_files, self.seq_len, self.sub_batch_size, self.next_character_prediction)
        
            # train_loader_super, valid_queue, num_classes = dp.data_preprocessing(self.train_directory, self.valid_directory, self.num_files, self.seq_len, self.super_batch_size, self.next_character_prediction)
        
            return train_loader_super, train_loader_sub, valid_loader, self.num_classes
    
     #def build_dataloader():
        # train_loader_sub, valid_loader, num_classes = dp.data_preprocessing(train_directory = train_directory, valid_directory = valid_directory, num_files= num_files,
        #   seq_size = seq_len, batch_size= sub_batch_size, next_character=next_character_prediction)
        
        # train_loader_super, valid_queue, num_classes = dp.data_preprocessing(train_directory = train_directory, valid_directory = valid_directory, num_files= num_files,
        #   seq_size = seq_len, batch_size= sub_batch_size, next_character=next_character_prediction)
        
     #   return train_loader_super, train_loader_sub, valid_loader
    

    def set_seed(self):
        np.random.seed(self.seed)
        cudnn.benchmark = True
        torch.manual_seed(self.seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(self.seed)



    def build_model(self, mask):
        
        multiplier, stem_multiplier = 4,3

        model = one_shot_model.RNNModelSearch(self.seq_len, self.dropouth, self.dropoutx,
                              self.init_channels, self.num_classes, self.layers, self.steps, multiplier, stem_multiplier,  
                              True, 0.2, None, self.task, 
                              mask).to(device)

        return model


    # archs, eval_archs = genotypes, eval_genotypes
    def train_and_eval(self, archs, eval_archs=None):
        """
        :param archs: archs sample by parallel BO
        :return: results list<genotype, top1acc>
        """
        # archs, eval_archs = genotypes, eval_genotypes
        self.genotypes = [eval(arch) if isinstance(arch, str) else arch for arch in archs] # speichert die genotypes jetzt als liste 
        self.eval_genos = None
        # ist None also überspringen
        if eval_archs != None:
            self.eval_genos = [eval(arch) if isinstance(arch, str) else arch for arch in eval_archs]
            self.genotypes = self.genotypes + self.eval_genos
        self.subnet_masks = [geno2mask(genotype) for genotype in self.genotypes]
        # genotype = genotypes[0]
        # subnet_masks = [geno2mask(genotype) for genotype in genotypes]
        # subnets_cnn = [for subnet[0] in subnets]
        cnn_masks = []
        rnn_masks = []
        for sub in self.subnet_masks:
            cnn_masks.append(sub[0])
            rnn_masks.append(sub[1])
        
        # supernet_mask = merge(subnet_cnn_masks, subnet_rnn_masks)

        supernet_mask = merge(cnn_masks, rnn_masks) # die 5 subnets zusammengefügt, damit er 1 großes supernet hat, welches aus den 100 init_samples/subarchitecturen gebildet wurde
        self.supernet_mask = list(supernet_mask)
        # supernet_mask = merge(cnn_masks, rnn_masks) # die 5 subnets zusammengefügt, damit er 1 großes supernet hat, welches aus den 100 init_samples/subarchitecturen gebildet wurde
    
        # len(subnet_masks)=5 und len(train_loader_sub)=391: erzeugt also eine list mit 391 elementen die eben immer 0,1,2,3,4 (wegen len(subnet_masks) sind)
        self.iterative_indices = list(islice(cycle(list(range(len(self.subnet_masks)))), len(self.train_loader_sub))) 
        # train_loader_sub wird als build_dataloader initialisiert und diese ist eine funktion 2 weiter oben definiert 
        # iterative_indices = list(islice(cycle(list(range(len(subnet_masks)))), len(train_loader_sub)))
        
        supernet = self.build_model(self.supernet_mask) # baut das model, gemäß der mask
        # supernet = build_model(supernet_mask) # baut das model, gemäß der mask

        logging.info("Training Super Model ...")
        logging.info("param size = %fMB", count_parameters_in_MB(supernet))
        
        conv = []
        rhn = []
        for name, param in supernet.named_parameters():
            #print(name)
            #if 'stem' or 'preprocess' or 'conv' or 'bn' or 'fc' in name:
           if 'rnns' in name:
               #print(name)
               rhn.append(param)
           #elif 'decoder' in name:
           else:
               #print(name)
               conv.append(param)
               
        self.conv = conv
        self.rhn = rhn
        
        optimizer = torch.optim.SGD([{'params':conv}, {'params':rhn}], lr=self.cnn_lr, weight_decay = self.cnn_weight_decay)
        optimizer.param_groups[0]['lr'] = self.cnn_lr
        optimizer.param_groups[0]['weight_decay'] = self.cnn_weight_decay
        optimizer.param_groups[0]['momentum'] = self.momentum
        optimizer.param_groups[1]['lr'] = self.rhn_lr
        optimizer.param_groups[1]['weight_decay'] = self.rhn_weight_decay
        
        
        #optimizer = torch.optim.SGD([{'params':conv}, {'params':rhn}], lr=cnn_lr, weight_decay = cnn_weight_decay)
        #optimizer.param_groups[0]['lr'] = cnn_lr
        #optimizer.param_groups[0]['weight_decay'] = cnn_weight_decay
        #optimizer.param_groups[0]['momentum'] = momentum
        #optimizer.param_groups[1]['lr'] = rhn_lr
        #optimizer.param_groups[1]['weight_decay'] = rhn_weight_decay
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.epochs))
        self.clip_params = [self.conv_clip, self.rhn_clip, self.clip]

        #params3 = []
        #for param in supernet.named_parameters():
        #    params3.append(param)

        # jetzt wird das modell erstmal für viele/1ne epochen mit den init_samples trainiert, um die gewichte des NN zu fitten
        # er trainiert erstmal für 1ne Epoche das supernetwork, indem er für jeden step ein child model sampled (haben 3000 steps also sampled er 3000 mal aus den 100/subnets)
        # danach validiert er dann mit dem supernetwork
        # und dann um ergebnisse zu bekommen für GCN, evaluiert er, indem er die 100/5 subnets jeweils für 3000 steps evaluiert
        for epoch in range(self.epochs): 
            # epoch=0
            logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
            supernet.drop_path_prob = self.drop_path_prob * epoch / self.epochs
            # supernet.drop_path_prob = drop_path_prob * epoch / epochs

            labels, predictions, train_loss = self.train(supernet, optimizer, supernet=False)
            
            labels = np.concatenate(labels)
            predictions = np.concatenate(predictions)
            
            if self.task == ('next_character_prediction'):
                acc = overall_acc(labels, predictions, self.task)
                logging.info('| epoch {:3d} | train acc {:5.2f}'.format(epoch, acc))
                logging.info('| epoch {:3d} | train loss {:5.2f}'.format(epoch, train_loss))
    
            else:
                f1 = overall_f1(labels, predictions, self.task)
                logging.info('| epoch {:3d} | train f1-score {:5.2f}'.format(epoch, f1))
                logging.info('| epoch {:3d} | train loss {:5.4f}'.format(epoch, train_loss))
               
                
            scheduler.step()
            
        logging.info("Evaluating subnets ...")
        
        # evaluate subnets with the weights trained with the supernet
        results = self.evaluate_subnets(supernet, self.subnet_masks, self.genotypes, self.eval_genos) # supernet ist das model, subnet_masks sind die 100/5 subnets einzeln abgespeichert

        return results

    def train_supernet(self, model, optimizer, epoch):
        # from generalNAS_tools.train_and_validate_HB import train, infer

        # labels, predictions, train_loss = train(self.train_loader_super, self.valid_loader, model, self.rhn, self.conv, self.criterion, optimizer, epoch, self.num_steps, self.clip_params, self.report_freq, self.beta, self.one_clip, task=self.task, mask=self.subnet_masks)

        labels, predictions, train_loss = self.train(model, optimizer, supernet=True) # train supernet for one epoch
        
        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        
        if self.task == ('next_character_prediction'):
            acc = overall_acc(labels, predictions, self.task)
            logging.info('| epoch {:3d} | train acc {:5.2f}'.format(epoch, acc))
            logging.info('| epoch {:3d} | train loss {:5.2f}'.format(epoch, train_loss))

        else:
            f1 = overall_f1(labels, predictions, self.task)
            logging.info('| epoch {:3d} | train f1-score {:5.2f}'.format(epoch, f1))
            logging.info('| epoch {:3d} | train loss {:5.4f}'.format(epoch, train_loss))
    
        if epoch == self.train_supernet_epochs - 1: # evaluate in last epoch
            # labels, predictions, valid_loss = infer(self.valid_loader, model, self.criterion, self.sub_batch_size, self.num_steps, self.report_freq, task=self.task, mask=self.subnet_masks)


            labels, predictions, valid_loss = self.evaluate(model, self.subnet_masks) # evaluate supernet 
            
            labels = np.concatenate(labels)
            predictions = np.concatenate(predictions)
            
            if self.task == ('next_character_prediction'):
                acc = overall_acc(labels, predictions, self.task)
                logging.info('| before resbn {:3d} | val acc {:5.2f}'.format(epoch, acc))
                logging.info('| before resbn {:3d} | val loss {:5.2f}'.format(epoch, valid_loss))
    
            else:
                f1 = overall_f1(labels, predictions, self.task)
                logging.info('| before resbn {:3d} | val f1-score {:5.2f}'.format(epoch, f1))
                logging.info('| before resbn {:3d} | val loss {:5.4f}'.format(epoch, valid_loss))
            #logging.info('Supernet valid %e %f %f', valid_loss, val_top1, val_top2)
            # update batchnorm operations with running mean: he resets the bn's
            set_running_statistics(model, self.train_loader_super, self.supernet_mask)
            # set_running_statistics(model, train_loader_super, supernet_mask)

            labels, predictions, valid_loss = self.evaluate(model, self.subnet_masks) # evaluate supernet 
            labels = np.concatenate(labels)
            predictions = np.concatenate(predictions)
            
            if self.task == ('next_character_prediction'):
                acc = overall_acc(labels, predictions, self.task)
                logging.info('| after resbn {:3d} | val acc {:5.2f}'.format(epoch, acc))
                logging.info('| after resbn {:3d} | val loss {:5.2f}'.format(epoch, valid_loss))
    
            else:
                f1 = overall_f1(labels, predictions, self.task)
                logging.info('| after resbn {:3d} | val f1-score {:5.2f}'.format(epoch, f1))
                logging.info('| after resbn {:3d} | val loss {:5.4f}'.format(epoch, valid_loss))

        copy_log_dir()

    def evaluate_subnets(self, supernet, subnet_masks, genotypes, eval_genos=None):
        results = []
        if eval_genos:
            genotypes = eval_genos
            subnet_masks = [geno2mask(geno) for geno in genotypes]
        with torch.no_grad():
            supernet.eval()
            supernet_copy = copy.deepcopy(supernet)
            # masks, genotypess = [], []
            # s=0
            i=1
            # top1 = [0.1,0.3,0.5] #0.2, 0.6]
            # er iteriert jetzt durch die einzelnen child models/den subnets (von welchen es 5/10 gibt), also evaluiert jedes einzelne
            for mask, genotype in zip(subnet_masks, genotypes): # er macht 5 iterationen für die 5 genotypes eben
                # results.append((genotype, top1[s])) # damit ich eine results liste bekomme
                # s+=1
                #                 
                # es ja nur BatchNorm betrifft und diese dann bei RHN's gar nicht aktiviert werden
                # obj, top1, top5 = self.evaluate(supernet_copy, mask) 
                set_running_statistics(supernet_copy, self.train_loader_sub, mask) # für mein search_space muss ich glaube ich nichts ändern, weil

                labels, predictions, valid_loss = self.evaluate(supernet_copy, mask) # evaluate supernet 
                
                # labels, predictions, valid_loss = evaluate(supernet_copy, subnet_masks[1], valid_loader, num_steps, criterion, task, report_freq) # evaluate supernet 

                labels = np.concatenate(labels)
                predictions = np.concatenate(predictions)
                
                if self.task == ('next_character_prediction'):
                    acc = overall_acc(labels, predictions, self.task)
                    #logging.info('| eval {:3d} | val acc {:5.2f}'.format(i, acc))
                    #logging.info('| eval {:3d} | val loss {:5.2f}'.format(i, valid_loss))
                    logging.info('%s th Arch %s valid %e %f',str(i), str(genotype.normal), valid_loss, acc)
                    results.append((genotype, acc))
        
                else:
                    f1 = overall_f1(labels, predictions, self.task)
                    #logging.info('| eval {:3d} | val f1-score {:5.2f}'.format(epoch, f1))
                    #logging.info('| eval {:3d} | val loss {:5.4f}'.format(epoch, valid_loss)) 

                    logging.info('%s th Arch %s valid %e %f',str(i), str(genotype.normal), valid_loss, f1)
                    # results.append((genotypes[1], f1))

                    results.append((genotype, f1))
                    
                copy_log_dir()
                i+=1
        return results
    
    
    
    def evaluate(self, model, mask):
        objs = AvgrageMeter()
             
        total_loss = 0
        labels = []
        predictions = []
        
        scores = nn.Softmax()
        
        with torch.no_grad():
        
            for step, (input, target) in enumerate(self.valid_loader):
    
                if step > self.num_steps:
                    break
                
                # input = input.transpose(1,2).float()
                input = input.to(device).float()
                batch_size = input.size(0)
        
                target = target.to(device)
                #target = torch.max(target, 1)[1]
                hidden = model.init_hidden(batch_size)#.to(device)  
                
                model.eval()
                # sub_mask = random.choice(mask) # subnet_mask enthält ja die adjacency matrizen zu 5 modellen

                logits, hidden = model(input, hidden, mask)
                #print(logits)
                loss = self.criterion(logits, target)
    
                objs.update(loss.data, batch_size)
                labels.append(target.detach().cpu().numpy())
                if self.task == "next_character_prediction":
                    predictions.append(scores(logits).detach().cpu().numpy())
                else:#if args.task == "TF_bindings"::
                    predictions.append(logits.detach().cpu().numpy())
        
                if step % self.report_freq == 0:
                    #logging.info('| step {:3d} | val obj {:5.2f} | '
                    #    'val acc {:8.2f}'.format(step,
                    #                               objs.avg, top1.avg))
                    logging.info('| step {:3d} | val obj {:5.2f}'.format(step, objs.avg))


        return labels, predictions, objs.avg.detach().cpu().numpy() # top1.avg, objs.avg
    

    # model, supernet = supernet, True
    # The super-network is trained by uniformly/randomly sampling from the
    # architectures of supernets. In each iteration, one sub-network (Ai, Xi) is randomly sampled from the super-network,
    def train(self, model, optimizer, supernet=False):
        objs = AvgrageMeter()
        total_loss = 0
        #start_time = time.time()
        scores = nn.Softmax()
        labels = []
        predictions = []
        
        model.train()
        iterative_indices = np.random.permutation(self.iterative_indices)
        train_loader = self.train_loader_super if supernet else self.train_loader_sub
        # train_loader = train_loader_super if supernet else train_loader_sub

        for step, (input, target) in enumerate(train_loader):
            
            if step > self.num_steps:
                break        
        
            input, target = input, target
           
            model.train()
            #input = input.transpose(1, 2).float()
    
            input = input.float().to(device)#.cuda()
            #target = torch.max(target, 1)[1]
            target = target.to(device)#.cuda(non_blocking=True)
            batch_size = input.size(0)
                        
            if self.mode == 'uniform': # in trainer_config wurde mode='random' übergeben
                mask = self.subnet_masks[iterative_indices[step]] if not supernet else self.supernet_mask
            else:
                # wir haben zum einen "self.supernet_mask" wo alle 5 subnets zusammengemerged wurden
                # und wir haben "self.subnet_masks" wo die 5 subnets (welche jeweils aus mask_cnn und mask_rnn besteht) noch einzeln gespeichert sind
                mask = random.choice(self.subnet_masks) # subnet_mask enthält ja die adjacency matrizen zu 5 modellen
                # er sampled hier 1 aus diesen 5 raus und diesen setzt er dann in model ein!!
                # er sampled aber in jedem step ein neues aus diesen 5 raus, d.h. bei 3000 steps hätten wir 3000 mal child_model gesampled
            #input = Variable(input).to(device)#.cuda()
            #target = Variable(target).to(device)#.cuda(async=True)
            
            optimizer.zero_grad()
            
            hidden = model.init_hidden(batch_size) 

            logits, hidden, rnn_hs, dropped_rnn_hs  = model(input, hidden, mask, return_h=True)
            
            raw_loss = self.criterion(logits, target)
            loss = raw_loss
           
            # Temporal Activation Regularization (slowness)
            #loss = loss + sum(self.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            #total_loss += raw_loss.data
            loss.backward()
          
            gc.collect()
            
            if self.one_clip == True:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.conv, self.conv_clip)
                torch.nn.utils.clip_grad_norm_(self.rhn, self.rhn_clip)
                
            optimizer.step()
                                
            objs.update(loss.data, batch_size)
            
            labels.append(target.detach().cpu().numpy())
            
            if self.task == "next_character_prediction":
                predictions.append(scores(logits).detach().cpu().numpy())
            else:
                predictions.append(logits.detach().cpu().numpy())
           
            #if step % report_freq == 0 and step > 0:
            
            #    logging.info('| step {:3d} | train obj {:5.2f}'.format(step, objs.avg))
                
    
        return labels, predictions, objs.avg.detach().cpu().numpy() # top1.avg, objs.avg
                
                
