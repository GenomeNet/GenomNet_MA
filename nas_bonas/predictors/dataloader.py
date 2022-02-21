"""Dataloader built for nasbench 101s"""
from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset
import pickle
from predictors.utils.gcn_utils import padzero, add_global_node

# er muss es mit Dataset initialiesieren welche oben importiert wird
# sample und maxize bekommt er in neual_net_gcn
 
# dataset = NasDataset(sample=__dataset, maxsize=maxsize)

class NasDataset(Dataset):
    """Nas bench 101 dataset."""

    def __init__(self, pickle_file=None, sample = None, samplenum = 10000, train = True, maxsize = 7):
        """
        Args:
            pickle_file (string): Path to the pickle file containing graphs (adj, label, metrics).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if(pickle_file):
            with open(pickle_file, "rb") as f:
                self.graphs = pickle.load(f)
                # self.graphs = random.sample(self.graphs,samplenum)
        elif sample is not None: # diese if-loop wird aktiviert, da sample=__dataset

            self.graphs = sample 
            # graphs=__dataset
        else:
            self.graphs = []
        self.train = train
        self.maxsize = maxsize


    def __len__(self):
        return len(self.graphs)

    """controls the dataloader to output different data during eval and trainings"""
    def swap_train_eval(self, ifTrain):
        self.train = ifTrain

    """process data """
    def __getitem__(self, graph_id):
        # print('get_item')
        adjacency_matrix_cnn = padzero(np.array(self.graphs[graph_id]['adjacency_matrix_cnn'],dtype=np.float32), ifAdj=True, maxsize=self.maxsize)
        # adjacency_matrix_cnn = padzero(np.array(graphs[graph_id]['adjacency_matrix_cnn'],dtype=np.float32), ifAdj=True, maxsize=self.maxsize)
        adjacency_matrix_cnn = add_global_node(adjacency_matrix_cnn, ifAdj = True)
        operations_cnn = padzero(np.array(self.graphs[graph_id]['operations_cnn'],dtype=np.float32), ifAdj=False, maxsize=self.maxsize)
        operations_cnn = add_global_node(operations_cnn, ifAdj = False)
        
        adjacency_matrix_rhn = padzero(np.array(self.graphs[graph_id]['adjacency_matrix_rhn'],dtype=np.float32), ifAdj=True, maxsize=self.maxsize)
        # adjacency_matrix_rhn = padzero(np.array(graphs[graph_id]['adjacency_matrix_rhn'],dtype=np.float32), ifAdj=True, maxsize=self.maxsize)
        adjacency_matrix_rhn = add_global_node(adjacency_matrix_rhn, ifAdj = True)
        operations_rhn = padzero(np.array(self.graphs[graph_id]['operations_rhn'],dtype=np.float32), ifAdj=False, maxsize=self.maxsize)
        operations_rhn = add_global_node(operations_rhn, ifAdj = False)
        
        accuracy = np.array(self.graphs[graph_id]['metrics'] ,dtype=np.float32)
        
        
        sample = {'adjacency_matrix_cnn': adjacency_matrix_cnn, 'operations_cnn': operations_cnn, 'adjacency_matrix_rhn': adjacency_matrix_rhn, 'operations_rhn': operations_rhn, 'accuracy': accuracy,}
        return sample
    
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        print('normalize')

        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        mx = np.dot(r_mat_inv,mx)
        return mx

    def stack_cell(self, mx, ifAdj):
        """stack adj or operation matrices into a cells"""
        print('stack_cell')

        if(ifAdj):
            big_matrix = np.zeros((mx.shape[0]*3,mx.shape[1]*3), dtype = np.float32)
            start_xpos = 0
            start_ypos = 0
            count = 0
            while count<3:
                for i in range(0,mx.shape[0]):
                    for j in range(0,mx.shape[1]):
                        big_matrix[start_xpos+i][start_ypos+j] = mx[i][j]
                if (count<2):
                    big_matrix[start_xpos+mx.shape[0]-1][start_ypos+mx.shape[1]]=1
                start_xpos += mx.shape[0]
                start_ypos += mx.shape[1]
                count+=1
        else:
            mx = np.column_stack((mx, np.zeros(mx.shape[0],dtype=np.float32)))
            big_matrix = np.concatenate((mx, mx), axis=0)
            big_matrix = np.concatenate((big_matrix, mx), axis=0)
        return big_matrix

    def stack_block(self, mx, ifAdj):
        """stack cells into a block"""
        print('stack_block')

        if(ifAdj):
            big_matrix = np.zeros((mx.shape[0]*3+2,mx.shape[1]*3+2), dtype = np.float32)
            start_xpos = 0
            start_ypos = 0
            count = 0
            while count<3:
                for i in range(0,mx.shape[0]):
                    for j in range(0,mx.shape[1]):
                        big_matrix[start_xpos+i][start_ypos+j] = mx[i][j]
                if(count<2):
                    big_matrix[start_xpos+mx.shape[0]-1][start_ypos+mx.shape[1]]=1
                    big_matrix[start_xpos+mx.shape[0]][start_ypos+mx.shape[1]+1]=1
                start_xpos += mx.shape[0]
                start_ypos += mx.shape[1]
                count+=1
        else:
            downsampling = np.array([[0,0,0,0,0,1]], dtype=np.float32)
            big_matrix = np.concatenate((mx, downsampling), axis=0)
            big_matrix = np.concatenate((big_matrix, mx), axis=0)
            big_matrix = np.concatenate((big_matrix, downsampling), axis=0)
            big_matrix = np.concatenate((big_matrix, mx), axis=0)


        return big_matrix


    def append_new_graph(self, new_graph):
        self.graphs += new_graph
    def get_dataset(self):
        return  self.graphs
