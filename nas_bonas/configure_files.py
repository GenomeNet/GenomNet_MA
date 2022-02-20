#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:47:48 2021

@author: amadeu
"""

import os

# alle scripte durchgehen, ob configure_files.py importiert wird

def get_io_config(taskname="supermodel_random_1"):
    
    # taskname = "supermodel_random_1"
    local_root_dir = "/home/ru85poq2/GenomNet_MA" # root working directory "/home/amadeu/anaconda3/envs/GenomNet_MA/BONAS" #
    local_data_dir = "/home/ru85poq2/data_BONAS" # data root "/home/amadeu/anaconda3/envs/GenomNet_MA/BONAS/data"
    results_dir = "trained_results"
    trained_pickle_file = "trained_models.pkl"
    trained_csv_file = "trained_models.csv"
    logfile = 'BOGCN_open_domain.log'
        
    io_config = dict(
         trained_pickle_file=os.path.join(local_root_dir, results_dir, taskname, trained_pickle_file),
         trained_csv_file=os.path.join(local_root_dir, results_dir, taskname, trained_csv_file),
    )
        
    # distributed = False
    
    return io_config, local_root_dir, local_data_dir, results_dir, taskname, local_data_dir, logfile


#taskname = "supermodel_random_1"
#local_root_dir = "/home/amadeu/Desktop/GenomNet_MA" # root working directory "/home/amadeu/anaconda3/envs/GenomNet_MA/BONAS" #
#local_data_dir = "/home/amadeu/data_BONAS" # data root "/home/amadeu/anaconda3/envs/GenomNet_MA/BONAS/data"
#results_dir = "trained_results"
#trained_pickle_file = "trained_models.pkl"
#trained_csv_file = "trained_models.csv"
#logfile = 'BOGCN_open_domain.log'
    
#io_config = dict(
#     trained_pickle_file=os.path.join(local_root_dir, results_dir, taskname, trained_pickle_file),
#     trained_csv_file=os.path.join(local_root_dir, results_dir, taskname, trained_csv_file),
#)
    
distributed = False
