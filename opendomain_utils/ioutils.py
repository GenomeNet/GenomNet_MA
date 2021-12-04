import os
import pickle
import shutil
from opendomain_utils.listdict import ListDict

# from BO_tools.configure_files import io_config, local_root_dir, local_data_dir, results_dir, taskname, \
#    local_data_dir, logfile

from BO_tools.configure_files import get_io_config
    
from opendomain_utils.genotypes import Genotype




# io_config, local_root_dir, local_data_dir, results_dir, taskname, local_data_dir, logfile = get_io_config(taskname="supermodel_random_1")

# trained_pickle_file = io_config['trained_pickle_file']
# trained_csv_file = io_config['trained_csv_file']

def create_dirs(local_root_dir, results_dir, taskname):
    
    print("creating dirs...")
    print(local_root_dir)
    local_result_dir = os.path.join(local_root_dir, results_dir, taskname)
    if not os.path.exists(local_result_dir):
        os.makedirs(local_result_dir)
    shutil.copy2(os.path.join(local_root_dir, "settings.py"), local_result_dir)


def get_trained_archs(trained_pickle_file):
    
    if os.path.isfile(trained_pickle_file):
        print("File exist")
        with open(trained_pickle_file, 'rb') as f:
            trained_list = pickle.load(f)
    else:
        print("File not exist")
        trained_list = []
        
    return trained_list


# datapoint = trained_archs
def update_trained_pickle(datapoint, trained_pickle_file):
    trained_list = get_trained_archs(trained_pickle_file)
    if isinstance(datapoint, dict): # datapoint=trained_archs wird jetzt einfach nur zu trained_list hinzugefügt
        trained_list.append(datapoint)
    elif isinstance(datapoint, list):
        trained_list.extend(datapoint)
    else:
        raise TypeError(f"datapoint is either a list or a dict, but got {type(datapoint)} instead")
    with open(trained_pickle_file, 'wb') as f:
        pickle.dump(trained_list, f)
    return trained_list

#trained_arch_list = trained_list  


def get_trained_csv(trained_csv_file):
    
    if os.path.isfile(trained_csv_file):
        print("File exist")
        trained_csv = ListDict.load_csv(path=trained_csv_file)
    else:
        print("File not exist")
        trained_csv = ListDict()
    return trained_csv


def update_trained_csv(datapoint, trained_csv_file):
    trained_csv = get_trained_csv(trained_csv_file)
    if isinstance(datapoint, dict):
        trained_csv.append(datapoint)
    elif isinstance(datapoint, list):
        trained_csv.extend(datapoint)
    else:
        raise TypeError(f"datapoint is either a list or a dict, but got {type(datapoint)} instead")
    trained_csv.to_csv(trained_csv_file)
    return trained_csv


def copy_log_dir():
    pass


def get_geno_hash(model_list):
    hashes = [model['hash'] for model in model_list]
    return hashes


def create_exp_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))
