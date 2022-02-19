#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:22:19 2021

@author: amadeu
"""

import time
from BO_tools import optimizer_gcn
import random
import argparse
import logging
from opendomain_utils.encode_and_train import encode_train, read_results
from opendomain_utils.transform_genotype import transform_Genotype
from data_generators.dynamic_generate import generate_archs

from opendomain_utils.ioutils import get_geno_hash, get_trained_csv, get_trained_archs, update_trained_csv, \
    update_trained_pickle, create_dirs
from BO_tools.configure_files import get_io_config

from samplers import EASampler, RandomSampler
import numpy as np
from BO_tools.trainer import Trainer
import os


class Runner(object):
    def __init__(self,
                 gcn_epochs=100,
                 gcn_lr=0.001,
                 loss_num=3,
                 generate_num=1000,
                 iterations=5,
                 sample_method="random",
                 training_cfg=None,
                 bo_sample_num=5,
                 mode="supernet",
                 if_init_samples=True,
                 init_num=10,
                 eval_submodel_path = None,
                 save = None,
                 save_dir=None
                 ):
        
        assert training_cfg is not None
        self.best_models = []

        self.mode = mode
        self.generate_num = generate_num
        self.max_acc = 0
        self.max_hash = 'none'
        self.max_genotype = 'none'
        self.gcn_lr = gcn_lr
        self.if_init_samples = if_init_samples
        self.bo_sample_num = bo_sample_num
        self.eval_submodel_path = eval_submodel_path
        self.loss_num = loss_num
        self.gcn_epochs = gcn_epochs
        self.iterations = iterations
        self.sample_method = sample_method
        self.save = save
        self.save_dir = save_dir

        io_config, local_root_dir, local_data_dir, results_dir, taskname, local_data_dir, logfile = get_io_config(self.save)
        # io_config, local_root_dir, local_data_dir, results_dir, taskname, local_data_dir, logfile = get_io_config(save)

        # create_dirs(local_root_dir, results_dir, self.save)

        self.trained_pickle_file = io_config['trained_pickle_file']
        self.trained_csv_file = io_config['trained_csv_file']
        
        self.trainer = Trainer(**training_cfg)
        
        self.train_init_samples(if_init_samples, init_num)
        self.build_optimizer()
        self.build_sampler()
      

    # die funktionen train_init_samples(), build_optimizer() und build_samples() werden oben in self initalisiert, also direkt ausgeführt und ergebnisse davon, stehen den anderen funktionen zur Verfügung
    # zuerst wird ja train_init_samples() initialisiert und dadurch werden in trained_arch_list file dann die Eigenschaften der 100/5 init architectures eingespeichert (weil er train_super_model ausführt, d.h. die 100/5 architecturen werden ausgeführt
    # und die ergebnisse/accuracies zusammen mit deren adjacency matritzen, genotypes etc. in trained_arch_list eingespeichert und damit das leere File jetzt ausgefüllt)
    
    # es werden 100 random supermodels trainiert
    # ist step 1. im Pseudocode, also D architectures damit wir GCN und BSR initialisieren können
  
    
    
    def train_init_samples(self, if_init_samples, init_num):
        if not if_init_samples:
            return
        print(self.trained_pickle_file)

        # original_graphs = generate_archs(generate_num=3) 
        # init_samples = random.sample(original_graphs, 2)
        original_graphs = generate_archs(generate_num=self.generate_num) # generiert glaube ich 10 random adj. matritzen 
        # gibt hier jetzt erstmal leere list zurück
        self.trained_arch_list = get_trained_archs(self.trained_pickle_file) # '/home/..../bonas_1/trained_models.pkl'
        logging.info("********************** training init samples **********************")
        # er sampled 100 mal diese 5 adjacency matritzen und trainiert dann dieses super_model (also werden insgesamt 100 dieser supermodels trainiert und trainer_archlist abgespeichert)
        while len(self.trained_arch_list) < init_num: # 100
            # er sampled random 5 adjacency matritzen mit seinen operationen
            # bo_sample_num = 3
            # init_samples bleibt gleiche wie original_graphs, nur eben weniger listenelemente, weil wir nur ein paar
            # original_graphs raussamplen
            init_samples = random.sample(original_graphs, self.bo_sample_num) # original_graphs von oben und bo_sample_num=5
            self.train_super_model(init_samples) # mit diesen trainiert er das super model
            self.trained_arch_list = get_trained_archs(self.trained_pickle_file)
            # trained_arch_list = get_trained_archs(trained_pickle_file)

            
        logging.info("********************** finished init training **********************")

    # glaube step 2. aus Pseudocode, bei dem mit den D architectures aus train_init_samples() das GCN und BSR initialisiert wird
    def build_optimizer(self):
        # builder BO optimizer
        self.trained_arch_list = get_trained_archs(self.trained_pickle_file)
        # Optimizer wird initialisiert, d.h. nur die funktion process_data() wird ausgeführt, wo A und X zu trained_arch_list gebildet wird
        self.optimizer = optimizer_gcn.Optimizer(self.trained_arch_list, train_epoch=self.gcn_epochs,
                                                 lr=self.gcn_lr, lossnum=self.loss_num)
        # optimizer = optimizer_gcn.Optimizer(trained_arch_list, train_epoch=gcn_epochs,
        #                                       lr=gcn_lr, lossnum=loss_num)
        # mit dem initialisierten optimizer (wo zu trained_arch_list schon X und A gebildet wurde in process_data() -> als __dataset abgespeichert)
        self.optimizer.train()

    # können ab sofort in anderen funktionen self.sampler ziehen und bekommen "random" sampler dafür (weil oben in sample_method so übergeben)
    # sampler = RandomSampler(generate_num=self.generate_num)
    def build_sampler(self):
        # build sampler
        # TODO: later add RL controller if needed
        if self.sample_method == 'random':
            print(self.generate_num)
            self.sampler = RandomSampler(generate_num=self.generate_num)
            # sampler = RandomSampler(generate_num=2)

        elif self.sample_method == 'ea':
            self.sampler = EASampler(self.trained_arch_list)
            # sampler = EASampler(trained_arch_list)
        else:
            raise NotImplementedError(f"{self.sample_method} is not a valid sampler type,"
                                      f"currently only support EA and Random")

    def train_one_model(self, data_point):
        
        # :param data_point: an architecture[adj,ops]
        # :return:trained acc result
        # trains an architecture and saves result to csv and pickle file
        
        self.trained_arch_list = get_trained_archs(self.trained_pickle_file)
        trained_models = get_geno_hash(self.trained_arch_list)
        adj, ops = data_point['adjacency_matrix'], data_point['operations']
        genohash = str(hash(str(transform_Genotype(adj, ops))))
        if genohash not in trained_models:
            encode_train(adj=adj, obj=ops, save_path=genohash)
            results = read_results(genohash)
            data_point = {'adjacency_matrix': adj, "operations": ops,
                          "metrics": results[1],
                          "genotype": str(results[0]), "hash": genohash}
            self.trained_arch_list = update_trained_pickle(
                data_point, self.trained_pickle_file)
            update_trained_csv(
                dict(genotype=str(results[0]), hashstr=genohash, acc=results[1]))
            if hasattr(self, 'optimizer'):
                self.optimizer.update_data(self.trained_arch_list)
            if hasattr(self, 'sampler'):
                self.sampler.update_sampler(data_point, ifappend=True)
            if results[1] > self.max_acc:
                self.max_acc = results[1]
                self.max_hash = genohash


    # data_points = init_samples
    def train_super_model(self, data_points):
        '''
                :param data_points: list<dict<adj,ops>>
                :return:trained accs and hashes of subnets
                '''
        # macht aus adjacency_matrix eben 5 genotypes
        # genotypes = [transform_Genotype(data_point['adjacency_matrix'], data_point['operations']) for data_point in
        #             data_points]
        
        genotypes = [transform_Genotype(data_point['adjacency_matrix_cnn'], data_point['operations_cnn'], data_point['adjacency_matrix_rhn'], 
                    data_point['operations_rhn']) for data_point in data_points]
        
        eval_genotypes = None
        genohashs = [hash(str(genotype)) for genotype in genotypes]
        #for genotype in genotypes:
        #    gene = genotype
        # hash(str(gene))
        # The hash() method returns the hash value of an object if it has one.
        # Hash values are just integers that are used to compare dictionary keys during a dictionary lookup quickly
   
        eval_num = len(genohashs) # 5
        if self.eval_submodel_path: # ist None, also überspringen
            import pickle
            with open(self.eval_submodel_path, 'rb') as f:
                datapoints = pickle.load(f)
                eval_genotypes = [datapoint['genotype'] for datapoint in datapoints]
                eval_num = len(eval_genotypes)
                
        results = self.trainer.train_and_eval(genotypes, eval_genotypes)
        # results = []
        # results.append((genotypes[0], 0.23))
        # results.append((genotypes[1], 0.17))
        # results.append((genotypes[1], 0.22))

        accs = [result[1] for result in results] # because 0th element is genotype and 1th element is always acc
        # accs = [0.5, 0.1, 0.9]
        trained_archs = [] # wird aufgefüllt mit adj. Matrix, operations, genotype und accuray von den 100/5 subarchitekturen
        trained_csvs = []
        trained_datapoints = [] # wird ebenfalls mit eigenschaften der 100/5 subarchitekturen abgespeichert
        for i in range(eval_num):
            # i = 1
            trained_arch = {'adjacency_matrix_cnn': data_points[i]['adjacency_matrix_cnn'],
                            "operations_cnn": data_points[i]['operations_cnn'],
                            'adjacency_matrix_rhn': data_points[i]['adjacency_matrix_rhn'],
                            "operations_rhn": data_points[i]['operations_rhn'],
                            "metrics": accs[i],
                            # "genotype": str(genotypes[i]), "hash": genohashs[i]}
                            "genotype": genotypes[i], "hash": genohashs[i]}

            trained_csv = {
                "genotype": str(genotypes[i]), 'hashstr': genohashs[i], 'acc': accs[i]
            }
            trained_archs.append(trained_arch)
            trained_csvs.append(trained_csv)
            # if-schleife nur True wenn neues bestes model
            if accs[i] > self.max_acc: # max_acc wird als 0 initialisiert, aber jetzt überschrieben mit neuem max_acc
                self.max_acc = accs[i] # neue max_acc wird überschrieben -> bestes model 
                self.max_hash = genohashs[i] # genotype vom besten model wird überschrieben
                self.max_genotype = genotypes[i]
            
            #if accs[i] > max_acc: # am ende haben wir max_acc mit bestem model und max_hash mit bestem genotype-hash (was eine nummer zu unserem genotype zuordnet)
            #    max_acc = accs[i] # 
            #    max_hash = genohashs[i] # geno
            trained_datapoints.append(trained_arch)
            
        self.best_models.append([self.max_acc, self.max_genotype])  
        self.trained_arch_list = update_trained_pickle(
            trained_archs, self.trained_pickle_file)
        print('len_trained_arch_list')
        print(len(self.trained_arch_list))
        update_trained_csv(trained_csvs, self.trained_csv_file)
        # ich glaub jetzt wird GCN auf initial design gefittet
        if hasattr(self, 'optimizer'):
            self.optimizer.update_data(self.trained_arch_list) # mit dieser trained_arch_list werden die adjacency Matritzen (globales) und feature Matrix gebildet
            # optimizer.update_data(trained_arch_list)
        if hasattr(self, 'sampler'):
            # trained_datapoints = trained_archs
            self.sampler.update_sampler(self.trained_arch_list, ifappend=True)
            # sampler.update_sampler(trained_arch_list, ifappend=True)


    def sample(self):
        new_domain = self.sampler.sample(self.generate_num)
        return new_domain
    
    # def sample(sampler):
    #    new_domain = sampler.sample(10)
    #    return new_domain

    def run_single(self):
        selection_index = 0
        selection_size = 0
        for iteration in range(self.iterations):
            if selection_index == selection_size:
                new_domain = self.sample()
                logging.info("Update GCN!!")
                self.optimizer.retrain_NN()
                logging.info("Update LR!!")
                self.optimizer.retrain_LR()
                selected_points, pred_acc, selected_indices = self.optimizer.select_multiple(new_domain=new_domain,
                                                                                             cap=self.bo_sample_num)
                selection_size = len(selected_points)
                selection_index = 0
                # Retrain the neural network
            logging.info(f"selection {str(selection_index)}, pred_Acc:{pred_acc[selection_index]}")
            # fully train models!!!!
            self.train_one_model(selected_points[selection_index])
            logging.info(f"iter{iteration},current max: {self.max_acc}, max hash {self.max_hash}")
            selection_index += 1

    # damit fängt er an, weil in "BOGCN_opendomain.py" fängt er mit .run an und da mode="supernet" ist, fangen wir mit run_super() an
    # er initialisiert oben ja erstmal train_init_samples() und build_optimizer() und build_model_sampler() muss diese also ausführen zuerst; deshalb 
    # haben wir eben optimizer und trainer_arch_list etc., 
    def run_super(self):
        # diese for-loop ist die foor-loop aus 5.-9. vom Pseudocode
        for iteration in range(self.iterations): # wurde oben mit 1000 übergeben aber in settings mit 0!!!
            # fully train super_model!!!!
            self.trained_arch_list = get_trained_archs(self.trained_pickle_file) # da train_init_samples() ja initialisiert wird, haben wir davon schon
            # trained_arch_list
            trained_models = get_geno_hash(self.trained_arch_list) 
            # trained_models = get_geno_hash(trained_arch_list)
            
            # step 4. vom Pseudocode, da sample candidate pool C from A with EA/Random
            new_domain = self.sample() # 2 weiter hoch gibt es die sample() funktion, welche sampler erhält und innerhalb der build_sampler() funktion definiert, weil hier wird sample_method aus argumente übergeben und dann haben wir eben Random oder EA sampler etc.
            # new_domain = sample(sampler) # 2 weiter hoch gibt es die sample() funktion, welche sampler erhält und innerhalb der build_sampler() funktion definiert, weil hier wird sample_method aus argumente übergeben und dann haben wir eben Random oder EA sampler etc.
            print('len_new_domain')
            print(len(new_domain))
            # jetzt macht er die steps 6.-10. weil er zuerst embedding mit GCN macht, dann mean und variance mit BSR und dann UCB
            # dann bestimmt er M top-k candidates, was hier selected_points ist!!!
            # bisher hat diese zeile nicht funktioniert, wegen Zeile 178 "sig ^ 2" in optimizer_gcn.py weil sig nan; aber mit integer klappt es dann
            start = time.time()
            selected_points, pred_acc, selected_indices = self.optimizer.select_multiple_unique(new_domain=new_domain,
                                                                                                cap=self.bo_sample_num, trained_models=trained_models)
            print('pred_acc')
            print(pred_acc)
            end = time.time()
            print('selection_time')
            print(end - start)
            # selected_points, pred_acc, selected_indices = optimizer.select_multiple_unique(new_domain=new_domain,
            #                                                                                     cap=bo_sample_num, trained_models=trained_models)
            # step 11.-12 of pseudocode: train the M top-k candidates with weight sharing                                                                                
            self.train_super_model(selected_points)
            logging.info(f"iter{iteration},current max: {self.max_acc}, max hash {self.max_hash}")
            logging.info("Update GCN!!")
            # jetzt macht er er step 13. weil er CGN und BSR updated mit diesen neuen Datenpunkten
            # self.optimizer.gcn
            start = time.time()
            self.optimizer.retrain_NN()
            end = time.time()
            print('gcn_update_time')
            print(end - start)

            # self.optimizer.gcn
            logging.info("Update LR!!")
            start = time.time()
            self.optimizer.retrain_LR()
            end = time.time()
            print('bsr_update_time')
            print(end - start)
            
        print(self.best_models)
        
        
        genotype_file = 'bonas_geno-{}'.format(self.save)
        np.save(os.path.join(self.save_dir, genotype_file), self.best_models[-1][-1])
    
        self.trained_arch_list = get_trained_archs(self.trained_pickle_file)
        samples_acc = []
        for i, sample in enumerate(self.trained_arch_list):
            samples_acc.append(sample['metrics'])

        samples_acc_file = 'samples_acc-{}'.format(self.save)
        np.save(os.path.join(self.save_dir, samples_acc_file), samples_acc)
    
        print(self.best_models[-1][-1])
        #np.save(os.path.join(self.save_dir, acc_train_file), train_acc)

        #time_file = 'time-{}'.format(self.save)
        #np.save(os.path.join(self.save_dir, time_file), time_per_epoch)

        # safe valid data
        #validloss_file = 'valid_loss-{}'.format(self.save)
        #np.save(os.path.join(self.save_dir, validloss_file), valid_losses)

        #acc_valid_file = 'acc_valid-{}'.format(self.save)
        #np.save(os.path.join(self.save_dir, acc_valid_file), valid_acc)
    
        

    def run(self):
        if self.mode == "supernet":
            self.run_super()
        elif self.mode == "singlenet":
            self.run_single()
        else:
            raise NotImplementedError
