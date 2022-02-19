#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 17:58:01 2021

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

from opendomain_utils.ioutils import copy_log_dir
from opendomain_utils.genotypes import Genotype
from settings import distributed

DataParallel = torch.nn.parallel.DistributedDataParallel if distributed else torch.nn.DataParallel
from itertools import cycle, islice
from opendomain_utils.bn_utils import set_running_statistics
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
from super_model import geno2mask, merge
from model_search import RNNModelSearch
import data_preprocessing as dp
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class Trainer:
    def __init__(self,
                 train_supernet_epochs=5,
                 data_path='data',
                 super_batch_size=2,
                 sub_batch_size=2,
                 learning_rate=0.1,
                 momentum=0.9,
                 weight_decay=3e-4,
                 report_freq=50,
                 epochs=50,
                 init_channels=36,
                 layers=20,
                 drop_path_prob=0.2,
                 seed=0,
                 grad_clip=5,
                 parallel = False,
                 mode='uniform',
                 train_directory = '/home/amadeu/Downloads/genomicData/train',
                 valid_directory = '/home/amadeu/Downloads/genomicData/validation',
                 num_files_train = 3,
                 num_files_valid = 3,
                 seq_len = 150,
                 num_steps = 1000,
                 dropout = 0.75,
                 dropouth = 0.25,
                 dropoutx = 0.75,
                 dropouti = 0.2,
                 dropoute = 0,
                 steps = 4,
                 multiplier = 4, 
                 stem_multiplier = 3 
                 ):
        self.parallel = parallel
        self.train_supernet_epochs = train_supernet_epochs
        self.data_path = data_path
        self.super_batch_size = super_batch_size
        self.sub_batch_size = sub_batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.mode = mode
        self.weight_decay = weight_decay
        self.report_freq = report_freq
        self.epochs = epochs
        self.init_channels = init_channels
        self.layers = layers
        self.drop_path_prob = drop_path_prob
        self.seed = seed
        self.grad_clip = grad_clip
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.train_directory = train_directory
        self.valid_directory = valid_directory
        self.num_files_train = num_files_train
        self.num_files_valid = num_files_valid
        self.seq_len = seq_len
        self.num_steps = num_steps
        self.dropout = dropout
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.steps = steps
        self.multiplier = multiplier
        self.stem_multiplier = stem_multiplier
        self.build_dataloader()


        self.train_loader_super, self.train_loader_sub, self.valid_loader, self.num_classes = self.build_dataloader()
        # train_loader_super, train_loader_sub, valid_loader = build_dataloader()



    
    
    # only difference between train_loader_sub and train_loader_super is different batch_size, and one is for the supermodel and the
    # other for the subnets
    
    def build_dataloader(self):
        train_loader_sub, valid_loader, num_classes = dp.data_preprocessing(self.train_directory, self.valid_directory, self.num_files_train, self.num_files_valid, self.seq_len, self.sub_batch_size)
        
        train_loader_super, valid_queue, num_classes = dp.data_preprocessing(self.train_directory, self.valid_directory, self.num_files_train, self.num_files_valid, self.seq_len, self.super_batch_size)
        
        return train_loader_super, train_loader_sub, valid_loader, num_classes
    
    
    
     #def build_dataloader():
     #   train_loader_sub, valid_loader, num_classes = dp.data_preprocessing(train_directory = train_directory, valid_directory = valid_directory, num_files_train= num_files_train,
     #     num_files_valid= num_files_valid, seq_size = seq_len, batch_size= sub_batch_size)
       
     #   train_loader_super, valid_queue, num_classes = dp.data_preprocessing(train_directory = train_directory, valid_directory = valid_directory, num_files_train= num_files_train,
     #     num_files_valid=num_files_valid, seq_size = seq_len, batch_size=super_batch_size)
        
     #   return train_loader_super, train_loader_sub, valid_loader
    

    def set_seed(self):
        np.random.seed(self.seed)
        cudnn.benchmark = True
        torch.manual_seed(self.seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(self.seed)




    def build_model(self, mask):
        model = RNNModelSearch(self.seq_len, self.dropout, self.dropouth, self.dropoutx, self.dropouti, self.dropoute,
                        self.init_channels, self.num_classes, self.layers, self.steps, self.multiplier,
                        self.stem_multiplier, True, 0.2, None, mask)
        #if self.parallel:
        #    if distributed:
        #        model = DataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
        #    else:
        #        model = DataParallel(model.cuda())
        #else:
        #    model = model.cuda()
        return model
    
    
    
    #seq_len, dropout, dropouth, dropoutx, dropouti, dropoute, init_channels, num_classes, layers, steps, multiplier, stem_multiplier = 10, 0.75, 0.25, 0.75, 0.2, 0, 16, 4, 3, 4, 4, 3
    #criterion = nn.CrossEntropyLoss().to(device)
    
    #def build_model(mask):
    #    model = RNNModelSearch(seq_len, dropout, dropouth, dropoutx, dropouti, dropoute,
    #                    init_channels, num_classes, layers, steps, multiplier,
    #                    stem_multiplier, True, 0.2, None,
    #                    mask)
        #if self.parallel:
        #    if distributed:
        #        model = DataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
        #    else:
        #        model = DataParallel(model.cuda())
        #else:
        #    model = model.cuda()
    #    return model
    
    #model = RNNModelSearch(seq_len, dropout, dropouth, dropoutx, dropouti, dropoute,
    #                    init_channels, num_classes, layers, steps, multiplier,
    #                    stem_multiplier, True, 0.2, None,
    #                    mask)
    
    #def build_model(mask):
    #    model = Network(init_channels, CIFAR_CLASSES, layers, mask=mask)
    #    if parallel:
    #        if distributed:
    #            model = DataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
    #        else:
    #            model = DataParallel(model.cuda())
    #    else:
    #        model = model.cuda()
    #    return model


    # archs = genotypes
    def train_and_eval(self, archs, eval_archs=None):
        """
        :param archs: archs sample by parallel BO
        :return: results list<genotype, top1acc>
        """
        # archs, eval_archs = genotypes, eval_genotypes
        self.genotypes = [eval(arch) if isinstance(arch, str) else arch for arch in archs] # speichert die genotypes jetzt als liste 
        self.eval_genos = None
        # ist None also überspringen
        #if eval_archs != None:
        self.eval_genos = [eval(arch) if isinstance(arch, str) else arch for arch in eval_archs]
             #self.genotypes = self.genotypes + self.eval_genos
        self.subnet_masks = [geno2mask(genotype) for genotype in self.genotypes] # transform genotypes into alpha matrix
        # genotype = genotypes[0]
        # subnet_masks = [geno2mask(genotype) for genotype in genotypes]
        # subnets_cnn = [for subnet[0] in subnets]
        cnn_masks = []
        rnn_masks = []
        for cnn_sub in self.subnet_masks:
            cnn_masks.append(cnn_sub[0])
        for rnn_sub in self.subnet_masks:
            rnn_masks.append(rnn_sub[1])
        
        # diese stelle müsste ich entfernen, wenn ich random search ohne WS haben will
        self.supernet_mask = merge(cnn_masks, rnn_masks) # die 5 subnets zusammengefügt, damit er 1 großes supernet hat, welches aus den 100 init_samples/subarchitecturen gebildet wurde
        
        # len(subnet_masks)=5 und len(train_loader_sub)=391: erzeugt also eine list mit 391 elementen die eben immer 0,1,2,3,4 (wegen len(subnet_masks) sind)
        self.iterative_indices = list(islice(cycle(list(range(len(self.subnet_masks)))), len(self.train_loader_sub))) 
        # train_loader_sub wird als build_dataloader initialisiert und diese ist eine funktion 2 weiter oben definiert 
        # iterative_indices = list(islice(cycle(list(range(len(subnet_masks)))), len(train_loader_sub)))
        
        super_model = self.build_model(self.supernet_mask) # baut das model, gemäß der mask
        # supernet = build_model(supernet_mask) # baut das model, gemäß der mask

        # am ende haben wir ähnliches one-shot model wie in 
        # init_channels, layers = 36,20
        # supernet = Network(init_channels, CIFAR_CLASSES, layers, mask=supernet_mask)

        logging.info("Training Super Model ...")
        logging.info("param size = %fMB", utils.count_parameters_in_MB(supernet))
        optimizer = torch.optim.SGD(
            super_model.parameters(), # er nimmt also nur die parameters vom "kleinen supernet" welches nur 100 subarchitectures beinhaltet
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # wird nur für 1ne Epoche trainiert
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.epochs))
    
        #params3 = []
        #for param in supernet.named_parameters():
        #    params3.append(param)

        # jetzt wird das modell erstmal für viele/1ne epochen mit den init_samples trainiert, um die gewichte des NN zu fitten
        # er trainiert erstmal für 1ne Epoche das supernetwork, indem er für jeden step ein child model sampled (haben 3000 steps also sampled er 3000 mal aus den 100/subnets)
        # danach validiert er dann mit dem supernetwork
        # und dann um ergebnisse zu bekommen für GCN, evaluiert er, indem er die 100/5 subnets jeweils für 3000 steps evaluiert
        for epoch in range(self.epochs): # in trainer_config wurde nur 1 epoche übergen, also jetzt nur 1ne iteration mit epoch=0
            logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
            supernet.drop_path_prob = self.drop_path_prob * epoch / self.epochs
            # supernet.drop_path_prob = drop_path_prob * epoch / epochs

            if epoch in range(self.train_supernet_epochs): # es wurde auch nur 1 supernet_epoche übergeben; weil range(0,1) bedeutet es wird nur 1ne iteration mit 0 gemaccht
                self.train_supernet(supernet, optimizer, epoch) # beides 0, also wird erste condition ausgeführt, wo zuerst ein
                # model trainiert wird aber nur mit 1nem gesampleten subnet (von insgesamt 5) aber evaluiert wird dann auf allen
                # er trainiert also auf 1ne subnet, aber evaluiert auf dem supernet, welches aus 5 subnets gemerged wurde
            else:
                self.train(supernet, optimizer, supernet=False)
               
                
            scheduler.step()
        logging.info("Evaluating subnets ...")
        
        # mit den angepassten gewichten werden jetzt die init_samples evaluiert umd val_accuracies zu bestimmen, dann haben wir
        # nämlich unsere vollständigen Beobachtungen mit labels für surrogate/GCN zu fitten
        results = self.evaluate_subnets(supernet, self.subnet_masks, self.genotypes, self.eval_genos) # supernet ist das model, subnet_masks sind die 100/5 subnets einzeln abgespeichert

        return results

    def train_supernet(self, model, optimizer, epoch):
        # macht zuerst train wo er supermodel trainiert und danach
        # evaluate um einzelne submodels zu evaluieren
        self.train(model, optimizer, supernet=True)
        if epoch == self.train_supernet_epochs - 1:
            # evaluiert mit großem supernet welche alle 100/5 subnets enthält
            val_obj, val_top1, val_top2 = self.evaluate(model, self.supernet_mask)
            logging.info('Supernet valid %e %f %f', val_obj, val_top1, val_top2)
            # jetzt datet er die batchnorm operations iwie mit running mean up: he resets the bn's
            set_running_statistics(model, train_loader_super, supernet_mask)
            # mit den upgedateten batchnorm operations validiert er nochmal mit selben daten
            val_obj, val_top1, val_top2 = self.evaluate(model, self.supernet_mask)
            logging.info('After resetbn Supernet valid %e %f %f', val_obj, val_top1, val_top5)

        copy_log_dir()

    def evaluate_subnets(self, supernet, subnet_masks, genotypes, eval_genos=None):
        results = []
       
        with torch.no_grad():
            supernet.eval()
            supernet_copy = copy.deepcopy(supernet)
            i=1
            # masks, genotypess = [], []
            # s=0
            # top1 = [0.1,0.3,0.5] #0.2, 0.6]
            # er iteriert jetzt durch die einzelnen child models/den subnets (von welchen es 5/10 gibt), also evaluiert jedes einzelne
            for mask, genotype in zip(subnet_masks, genotypes): # er macht 5 iterationen für die 5 genotypes eben
                #results.append((genotype, top1[s])) # damit ich eine results liste bekomme
                #s+=1
                
                set_running_statistics(supernet_copy, self.train_loader_sub, mask) # für mein search_space muss ich glaube ich nichts ändern, weil
                # es ja nur BatchNorm betrifft und diese dann bei RHN's gar nicht aktiviert werden
                obj, top1, top5 = self.evaluate(supernet_copy, mask) 
                logging.info('%s th Arch %s valid %e %f %f',str(i), str(genotype.normal), obj, top1, top5)
                # 
                results.append((genotype, top1))
                copy_log_dir()
                i+=1
        return results
    
    def evaluate(self, model, mask):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top2 = utils.AvgrageMeter()
        model.eval()
        with torch.no_grad():
            for step, (input, target) in enumerate(self.valid_loader):
                
                input = Variable(input).to(device)#.cuda()
                target = Variable(target).to(device)#.cuda(async=True)
                
                if step > self.num_steps:
                    break
                #input, target = input, target
               
                input = input.transpose(1, 2).float()
        
                target = torch.max(target, 1)[1]
                batch_size = input.size(0)
    
                hidden = model.init_hidden(batch_size)
                
                logits, hidden, rnn_hs, dropped_rnn_hs  = model(input, hidden, mask)
                
                loss = self.criterion(logits, target)
                prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
                objs.update(loss.data.item(), batch_size)
                top1.update(prec1.data.item(), batch_size)
                top2.update(prec2.data.item(), batch_size)
        return objs.avg, top1.avg / 100, top2.avg / 100

    # model, supernet = supernet, True
    # The super-network is trained by uniformly/randomly sampling from the
    # architectures of supernets. In each iteration, one sub-network (Ai, Xi) is randomly sampled from the super-network,
    def train(self, model, optimizer, supernet=False):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top2 = utils.AvgrageMeter()
        # model = supernet
        model.train()
        iterative_indices = np.random.permutation(self.iterative_indices)
        train_loader = self.train_loader_super if supernet else self.train_loader_sub
        # train_loader = train_loader_super if supernet else train_loader_sub
        
        #params2 = []
        #for param in model.named_parameters():
        #    params2.append(param)

        for step, (input, target) in enumerate(train_loader):
            
            input = Variable(input).to(device)#.cuda()
            target = Variable(target).to(device)#.
            
            if step > self.num_steps:
                break
            #input, target = input, target
           
            input = input.transpose(1, 2).float()
    
            target = torch.max(target, 1)[1]
            batch_size = input.size(0)

                
            hidden = model.init_hidden(batch_size)
            #hidden_valid = model.init_hidden(args.batch_size
            
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
            
            logits, hidden, rnn_hs, dropped_rnn_hs  = model(input, hidden, mask)
            
            # criterion = nn.CrossEntropyLoss()
            loss = self.criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()

            prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
            objs.update(loss.data.item(), batch_size)
            top1.update(prec1.data.item(), batch_size)
            top2.update(prec2.data.item(), batch_size)
            if step % self.report_freq == 0:
                logging.info('supernet train %03d %e %f %f', step, objs.avg, top1.avg / 100, top2.avg / 100)
                copy_log_dir()
