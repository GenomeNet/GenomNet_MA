#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:23:20 2021

@author: amadeu
"""

import os, sys, glob

import logging
# from BO_tools.configure_files import local_root_dir, local_data_dir, logfile, taskname, results_dir
from BO_tools.runner import Runner
import argparse
from generalNAS_tools import utils




parser = argparse.ArgumentParser("search")
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu')
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
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--cnn_lr', type=float, default=0.025, help='learning rate for CNN part')
parser.add_argument('--cnn_weight_decay', type=float, default=3e-4, help='weight decay for CNN part')
parser.add_argument('--rhn_lr', type=float, default=8, help='learning rate for RHN part')
parser.add_argument('--rhn_weight_decay', type=float, default=5e-7, help='weight decay for RHN part')
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--num_steps', type=int, default=2, help='number of iterations per epoch')
parser.add_argument('--next_character_prediction', type=bool, default=True, help='task of model')
parser.add_argument('--one_clip', type=bool, default=True)
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--conv_clip', type=float, default=5, help='gradient clipping of convs')
parser.add_argument('--rhn_clip', type=float, default=0.25, help='gradient clipping of lstms')
parser.add_argument('--seq_size', type=int, default=1000, help='sequence length')

parser.add_argument('--gcn_epochs', type=int, default=2, help='epochs of GCN surrogate')
parser.add_argument('--gcn_lr', type=float, default=0.001, help='learning rate for GCN surrogate')
parser.add_argument('--loss_num', type=int, default=3, help='number of losses')
parser.add_argument('--generate_num', type=int, default=10, help='number of sampled child models in each BONAS iteration')
parser.add_argument('--iterations', type=int, default=3, help='iterations of BONAS algorithm')
parser.add_argument('--bo_sample_num', type=int, default=10, help='number of child models, which are run in each BONAS iteration')
parser.add_argument('--sample_method', type=str, default='random', help='sample method for child models')
parser.add_argument('--if_init_samples', type=bool, default=True)
parser.add_argument('--init_num', type=int, default=10, help='number of child models for initial design')

parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')

parser.add_argument('--train_supernet_epochs', type=int, default=1, help='epochs of supernet')
parser.add_argument('--super_batch_size', type=int, default=2, help='batch_size for supernet')
parser.add_argument('--sub_batch_size', type=int, default=2, help='batch_size for subnets')

parser.add_argument('--report_freq', type=int, default=1000, help='report frequency')
parser.add_argument('--epochs', type=int, default=1, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')

parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='report frequency')
parser.add_argument('--mode', type=str, default='random', help='sample method for child models')

parser.add_argument('--save', type=str,  default='search',
                    help='name to save the labels and predicitons')
parser.add_argument('--save_dir', type=str,  default='test_search',
                    help='path to save the labels and predicitons')

#parser.add_argument('--taskname', type=str, default='bonas_model', help='name of task')
#parser.add_argument('--local_root_dir', type=str, default='/home/amadeu/', help='directory, where results are saved')
#parser.add_argument('--local_data_dir', type=str, default='/home/amadeu/data_BONAS', help='directory, where data ist stored')
#parser.add_argument('--results_dir', type=str, default='trained_results', help='directory, where results are saved')
args = parser.parse_args()


#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#logging.basicConfig(filename=os.path.join(local_root_dir, results_dir, taskname, str(args.gpu)+logfile), filemode='w', level=logging.INFO,
#                    format='%(asctime)s : %(levelname)s  %(message)s', datefmt='%Y-%m-%d %A %H:%M:%S')
#os.environ["PYTHONHASHSEED"] = "0"
#console = logging.StreamHandler()
#console.setLevel(logging.INFO)
#formatter = logging.Formatter('%(asctime)s : %(levelname)s  %(message)s')
#console.setFormatter(formatter)
#logging.getLogger().addHandler(console)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
#logging.getLogger().addHandler(fh)

logging = logging.getLogger(__name__)


from BO_tools.configure_files import get_io_config
io_config, local_root_dir, local_data_dir, results_dir, taskname, local_data_dir, logfile = get_io_config(args.save)




# initial_designs: 3h per epoch (mit 100 initial archs): -> 1 Tag für training von initial_design (wenn für 10 epochen trainiert)
# er random_sampled 100 aber aus denen trainiert er ebenfalls 100
# "generate_num" ist immer anzahl an archs, und "bo_sample_num" ist anzahl an archs die trainiert werden sollen (bei initial_design durch randomsampling später durch ucb-score)


### mein vorschlag:
# bo_sample_nun=100 so wie er; iterations = 15; num_epochs=100 mit supernetepochs=10 so wie er, weil wir in jeder epoche auch neue sub_architektur random samplen, also er trainiert immer nur mit 1nem subnet
# bei epochs 0:10 ist supernet=True, d.h. er hat nur anderen train_loader und ab 10 bis 100 trainiert er mit supernet=False
# dann würden wir 10iterations*100epochs*1hperepoch=300h, also 12.5 Tage nur zum trainieren
# 15iterations*1epochs*100subnets*xh=... also ... Tage zum evaluieren der subnets
# und dann haben wir auch noch 100GCN epochs, von dem GCN-surrogate
# außerdem noch 1000 predictions von dem GCN-surrogate
# nur über iterations kürzen, weil sonst zerstören wir zu viel vom algorithms!!!!

### abchecken indem ich am ende auf cluster laufen lasse:
# wie lange für 100 samples für 100 epochen zu trainieren (einfach mein "richtiger run laufen lassen und durch meine prints nach jeder epoche vom training sehe ich ja wie lange dauert)
# -> er braucht 1.5h pro epoche: 1.5*100=150h, was 6 Tage sind(auf MCML Cluster)

# wie lange für die 100 GCN epochs (time wird nach jedem optimizer.update geprintet)
# wie lange für 1000 GCN predictions (time wird nach jedem selected_unique geprintet)
# wie wird alles abgespeichert, bzw. läuft mein algo durch? (mit weniger iterations, steps etc. mal durchlaufen lassen: brauche ja nur max_acc und max_genotype)
# datet er wirklich das GCN richtig ab? (einfach mein "richtiger run laufen lassen und durch meine prints sehe ich "len_trained_arch_list" bevor er im runner optimizer.update_data() macht)


# er definiert in build_optimizer() ja erstmal den optimizer (bei initialisierung wird nur _dataset definiert) und danach trainiert er, wo auch gcn definiert wird
# bei jeder iteration macht er train_super_model mit selected_points, bei dem am ende immer optimizer.update_data steht, was zu neuem self.adj_cnn führt auf Basis von "len_trained_arch_list" (sollte 100, dann 200 usw. sein)
# checken wie GCN trainiert wird (batch_size): anscheinend 128
# -> indem ich bei gcn_train_val die tensoren printe


# checken, ob er gewichte von gcn weight sharing zwischen iterationen macht! indem ich print(gcn.parameters()) oder so mache
# evtl. trainiert er immer wieder neu, aber mit allen also auch alten archs
# -> indem ich bei runner() len_trained_arch_list printe sehe ich ob alte und neue gemerged werden

# checken ob er wirklich neues train_adj nimmt (innerhalb von retrain_NN() print(self.train_adj) machen)
# prüfen, ob er wirklich immer trained_models updated
# -> indem ich bei retrain_NN() self.__train_dataset printe


# fragen in issues wegen EA-sampler


### genomicBONAS debuggen:
    
# teilweise kommt dieser error, weil wrsl. NaNs produziert werden. diese kommen immer nur bei evaluate! und er macht nur nach der letzten training epoche von train_supernet (also nach der 10ten) auch eine evaluate epoche!!
# evtl. ist er nach 10 epochen noch zu schlecht trainiert, also reicht aus länger zu trainieren!
# ist es wirklich wegen NaNs? laufen lassen und labels printen lassen
# hier wird der error gut erklärt: https://towardsdatascience.com/cuda-error-device-side-assert-triggered-c6ae1c8fa4c3 ; er schreibt es liegt meisten schon daran, wenn er outputs produziert, die nicht zwischen 0 und 1 liegen
# ich könnte mal einen test_run laufen lassen, nur mit 10 num_bo_samples aber mit 50 epochen trainieren lassen, und train_supernet_epochs nach 20 epochen
# einen run mit set_running_statistics und einen ohne
# evtl. habe ich irgendeinen error
# wobei andererseits auch bei HB-NAS die validation immer schlecht ist (der ja selben search-space besitzt): evtl. also doch ein error der wegen search-space kommt
# -> ich sehe schon, dass validation_loss immer bei 0.2 ist also kann irgendwas nicht stimmen
# könnte ich prüfen, indem einen run starte, von HB-NAS und in pretrain_epochs mal auch validation printen lasse! muss ja dann f1-score, ähnlich wie train f1-score rausgeben


search_config = dict(
    gcn_epochs=args.gcn_epochs,#100,
    gcn_lr=args.gcn_lr,
    loss_num=args.loss_num,
    generate_num=args.generate_num,#100,
    iterations=args.iterations,
    bo_sample_num=args.bo_sample_num,#100,
    sample_method=args.sample_method,
    if_init_samples=args.if_init_samples,
    init_num=args.init_num,#100,
    save = args.save,
    save_dir = args.save_dir
)


training_config = dict(
    train_supernet_epochs=args.train_supernet_epochs,
    data_path=os.path.join(local_data_dir, 'data'),
    super_batch_size=args.super_batch_size,#64,
    sub_batch_size=args.sub_batch_size,#128,
    
    cnn_lr=args.cnn_lr,
    cnn_weight_decay=args.cnn_weight_decay,
    rhn_lr=args.rhn_lr,
    rhn_weight_decay=args.rhn_weight_decay,
    momentum=args.momentum,
    
    report_freq=args.report_freq,
    epochs=args.epochs,
    init_channels=args.init_channels,#36,
    layers= args.layers,#20,
    drop_path_prob=args.drop_path_prob,
    seed=0,
    #grad_clip=args.clip,
    parallel=False,
    mode=args.mode,
    train_directory = args.train_directory,
    valid_directory = args.valid_directory,
    test_directory = args.test_directory,
    
    train_input_directory = args.train_input_directory,
    train_target_directory = args.train_target_directory,
    valid_input_directory = args.valid_input_directory,
    valid_target_directory = args.valid_target_directory,
    test_input_directory = args.test_input_directory,
    test_target_directory = args.test_target_directory,
    task = args.task,

    num_files = args.num_files,
    seq_len = args.seq_size,
    num_steps = args.num_steps,
    next_character_prediction=args.next_character_prediction,
    dropouth = args.dropouth,
    dropoutx = args.dropoutx,
    one_clip = args.one_clip,
    clip = args.clip,
    conv_clip = args.conv_clip,
    rhn_clip = args.rhn_clip,
    beta=args.beta
)




if __name__ == "__main__":
    runner = Runner(**search_config, training_cfg=training_config) # training_config und search_config wird beides von
    runner.run()