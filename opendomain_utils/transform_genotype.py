import numpy as np
from opendomain_utils.genotypes import Genotype, PRIMITIVES
# from GenomNet_MA.genomicNAS_algorithms.train_genomicBONAS import OPS_cnn, OPS_rnn
from generalNAS_tools.genotypes import OPS_cnn, OPS_rnn
from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT


# adj_cnn = data_points[0]["adjacency_matrix_cnn"]
# ops_cnn = data_points[0]["operations_cnn"]
# adj_rhn = data_points[0]["adjacency_matrix_rhn"]
# ops_rhn = data_points[0]["operations_rhn"]

def transform_Genotype(adj_cnn, ops_cnn, adj_rhn, ops_rhn):
    # delete row 2,4,6,8 (because we have two of each)
    adj_cnn = np.delete(adj_cnn, 2, axis=0)
    adj_cnn = np.delete(adj_cnn, 3, axis=0)
    adj_cnn = np.delete(adj_cnn, 4, axis=0)
    adj_cnn = np.delete(adj_cnn, 5, axis=0)
    cell_cnn = [
        # node 1
        (OPS_cnn[np.nonzero(ops_cnn[2])[0][0]], np.nonzero(adj_cnn[:, 2])[0][0]), # second row of ops_cnn correspond to first input operation of node1; second column of adj_cnn correspond to first input of node 1
        (OPS_cnn[np.nonzero(ops_cnn[3])[0][0]], np.nonzero(adj_cnn[:, 3])[0][0]), # third row of ops_cnn correspond to second input operation of node1; third column of adj_cnn correspond to second input of node 1
        # node 2
        (OPS_cnn[np.nonzero(ops_cnn[4])[0][0]], np.nonzero(adj_cnn[:, 4])[0][0]),
        (OPS_cnn[np.nonzero(ops_cnn[5])[0][0]], np.nonzero(adj_cnn[:, 5])[0][0]),
        # node 3
        (OPS_cnn[np.nonzero(ops_cnn[6])[0][0]], np.nonzero(adj_cnn[:, 6])[0][0]),
        (OPS_cnn[np.nonzero(ops_cnn[7])[0][0]], np.nonzero(adj_cnn[:, 7])[0][0]),
        # node 4
        (OPS_cnn[np.nonzero(ops_cnn[8])[0][0]], np.nonzero(adj_cnn[:, 8])[0][0]),
        (OPS_cnn[np.nonzero(ops_cnn[9])[0][0]], np.nonzero(adj_cnn[:, 9])[0][0]),
    ]
    
    
    cell_rnn = [ 
      
        (OPS_rnn[np.nonzero(ops_rhn[3])[0][0]], np.nonzero(adj_rhn[:, 3])[0][0]-2), # need to subtract 2, because intermediate node receives both input nodes
        (OPS_rnn[np.nonzero(ops_rhn[4])[0][0]], np.nonzero(adj_rhn[:, 4])[0][0]-2), 
        (OPS_rnn[np.nonzero(ops_rhn[5])[0][0]], np.nonzero(adj_rhn[:, 5])[0][0]-2),
        (OPS_rnn[np.nonzero(ops_rhn[6])[0][0]], np.nonzero(adj_rhn[:, 6])[0][0]-2),
        (OPS_rnn[np.nonzero(ops_rhn[7])[0][0]], np.nonzero(adj_rhn[:, 7])[0][0]-2),
        (OPS_rnn[np.nonzero(ops_rhn[8])[0][0]], np.nonzero(adj_rhn[:, 8])[0][0]-2),
        (OPS_rnn[np.nonzero(ops_rhn[9])[0][0]], np.nonzero(adj_rhn[:, 9])[0][0]-2),
        (OPS_rnn[np.nonzero(ops_rhn[10])[0][0]], np.nonzero(adj_rhn[:, 10])[0][0]-2),
    ]
    
    ft_model = Genotype(
        normal=cell_cnn,
        normal_concat=[2, 3, 4, 5],
        reduce=cell_cnn,
        reduce_concat=[2, 3, 4, 5],
        rnn = cell_rnn,
        rnn_concat = range(1,9)
    )
    return ft_model


# genotype = genotypes[1]
def transform_matrix(genotype):
    # cnn part
    normal = genotype.normal
    node_num = len(normal)+3
    adj_cnn = np.zeros((node_num, node_num))
    ops_cnn = np.zeros((node_num, len(OPS_cnn)))
    for i in range(len(normal)):
        op, connect = normal[i]
        if connect == 0 or connect==1:
            adj_cnn[connect][i+2] = 1
        else:
            adj_cnn[(connect-2)*2+2][i+2] = 1
            adj_cnn[(connect-2)*2+3][i+2] = 1
        ops_cnn[i+2][OPS_cnn.index(op)] = 1
    
    adj_cnn[2:-1, -1] = 1 # output node, connected to all 8 previous node
    ops_cnn[0:2, 0] = 1 # input operation
    ops_cnn[-1][-1] = 1 # output operation
    
     # rhn part
    rnn = genotype.rnn
    node_num = len(rnn)+4
    adj_rnn = np.zeros((node_num, node_num))
    ops_rnn = np.zeros((node_num, len(OPS_rnn)))
    for i in range(len(rnn)):
        op, connect = rnn[i]
        
        adj_rnn[connect+2][i+3] = 1
        
        ops_rnn[i+3][OPS_rnn.index(op)] = 1
    
    adj_rnn[3:-1, -1] = 1 # output node, connected to all 8 previous node
    adj_rnn[0:2, 2] = 1 # output node, connected to all 8 previous node

    ops_rnn[0:2, 0] = 1 # input operation
    ops_rnn[2, 1] = 1 # intermediate operation
    ops_rnn[-1][-1] = 1 # output operation
    return adj_cnn, ops_cnn, adj_rnn, ops_rnn

def geno_to_archs(genotypes, ei_scores=None):
    # print(genotypes)
    archs = []
    for i in range(len(genotypes)):
        # i = 0
        if isinstance(genotypes, str):
            adj_cnn, ops_cnn, adj_rnn, ops_rnn = transform_matrix(eval(genotypes[i]))
        else:
            adj_cnn, ops_cnn, adj_rnn, ops_rnn = transform_matrix(genotypes[i])
            
        if ei_scores:
            
            datapoint = {'adjacency_matrix_cnn': adj_cnn, 'operations_cnn': ops_cnn, 'adjacency_matrix_rhn': adj_rnn, 'operations_rhn': ops_rnn, 'metrics': ei_scores[i]}
       
        else:
            
            datapoint = {'adjacency_matrix_cnn': adj_cnn, 'operations_cnn': ops_cnn, 'adjacency_matrix_rhn': adj_rnn, 'operations_rhn': ops_rnn}
            
        archs.append(datapoint)
        
    return archs

if __name__ == '__main__':
    A = np.array([
        [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    ops = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
    ])
    geno = transform_Genotype(A, ops)
    print(geno.normal)
    # print(geno.ops)
    # print(transform_matrix(geno) - A)
    print(transform_matrix(geno)[1] - ops)
