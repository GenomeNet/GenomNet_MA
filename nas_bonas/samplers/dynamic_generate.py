

import random
import numpy as np
import time
from generalNAS_tools.genotypes import OPS_cnn, OPS_rnn


"""
This file is used for generating archs in the search space
"""



def generate_adj_rhn():
    ## Random choices for RHN ## 
    mat = np.zeros([12, 12])
    mat[:, 11] = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    a = random.choice([2,3])
    b = random.choice([2, 3, 4])
    c = random.choice([2, 3, 4, 5])
    d = random.choice([2, 3, 4, 5, 6])
    e = random.choice([2, 3, 4, 5, 6, 7])
    f = random.choice([2, 3, 4, 5, 6, 7, 8])
    g = random.choice([2, 3, 4, 5, 6, 7, 8, 9])
    
    mat[0, 2] = 1
    mat[1, 2] = 1

    mat[2, 3] = 1
    
    mat[a, 4] = 1
    mat[b, 5] = 1
    mat[c, 6] = 1
    mat[d, 7] = 1
    mat[e, 8] = 1
    mat[f, 9] = 1
    mat[g, 10] = 1
    
    return mat





# erzeugt glaube ich einfach nur eine random adj matrix 11x11
def generate_adj_cnn():
    mat = np.zeros([11, 11])
    mat[:, 10] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    a = random.choice([0, 1])
    b = random.choice([0, 1])
    c = random.choice([0, 1, [2,3]])
    d = random.choice([0, 1, [2,3]])
    e = random.choice([0, 1, [2,3], [4,5]])
    f = random.choice([0, 1, [2,3], [4,5]])
    g = random.choice([0, 1, [2,3], [4,5],[6,7]])
    h = random.choice([0, 1, [2,3], [4,5],[6,7]])
    mat[a, 2] = 1
    mat[b, 3] = 1
    mat[c, 4] = 1
    mat[d, 5] = 1
    mat[e, 6] = 1
    mat[f, 7] = 1
    mat[g, 8] = 1
    mat[h, 9] = 1
    return mat


def generate_ops_cnn():
    op_num = len(OPS_cnn)
    op_matrix = np.zeros((11, op_num))
#     op_matrix = np.zeros((11, 6))
    op_matrix[0][0] = 1
    op_matrix[1][0] = 1
    for i in range(8):
        idx = random.choice(list(range(1, op_num-1))) # exclude 'input_cnn' and 'output_cnn' as operation
        op_matrix[i + 2][idx] = 1
    op_matrix[10][-1] = 1
    return op_matrix



def generate_ops_rhn():
    op_num = len(OPS_rnn)
    op_matrix = np.zeros((12, op_num))
#     op_matrix = np.zeros((11, 6))
    op_matrix[0][0] = 1
    op_matrix[1][0] = 1
    op_matrix[2][1] = 1

    for i in range(8):
        idx = random.choice(list(range(2, op_num-1)))
        op_matrix[i + 3][idx] = 1
    op_matrix[11][-1] = 1
    return op_matrix




#test = generate_adj() # ergibt 11x11 matrix

# genereriert eine random feature matrix 11x10
def generate_ops():
    op_num_cnn = len(OPS_cnn)
    op_num_rnn = len(OPS_rnn)
    op_matrix = np.zeros((19, op_num_cnn+op_num_rnn)) # rows stand for nodes/edges
    ## first 2 rows receive 'input' as operation
    op_matrix[0][0] = 1 # 1th operation is 'input', cell_t-1 
    op_matrix[1][0] = 1 # 2th operation is 'input', weil cell_t-2
    # loop over 16 rows; 17th row can not do any operation as it is the last node of rnn and 18th node is the global node
    for i in range(8+8): # 0,1,2,...7; 8,9,10,11,12,13,14,15
        if i<8: # 0,1,2,3,4,5,6,7
            # choose randomly one of the 6 possible operation
            idx_cnn = random.choice(list(range(1, op_num_cnn))) # random choice except 0 which stands for 'input' operation
            # fängt bei 1 und nicht 0 an, weil 0 ist Inputoperation und die darf nicht gewählt werden
            op_matrix[i + 2][idx_cnn] = 1 # füge an spalten-stelle idx eine 1 ein, und Zeile ist 2,3,4,5,6,7,8 ()
        # Zeilen 10-18 bekommen random rnn operationen
        if i>=8: # 8,9,10,11,12,13,14,15
            # i=14
            idx_rnn = random.choice(list(range(0, op_num_rnn-1))) # random choice except 4 which stands for 'output' operation

            op_matrix[i + 2][idx_rnn+5] = 1 # füge an spalten-stelle idx eine 1 ein, und Zeile ist 2,3,4,5,6,7,8 ()

    #op_matrix[10][6] = 1 # weil letzte Zeile von CNN ist natürlich die Operation None: deshalb
    # müssen wir an letzter Zeile (10) von CNN zugreifen weil für output Node steht und dann eben an letzter spalte von dieser Zeile eine 1 setzen, 
    # weil letzte Spalte steht für operation output
    op_matrix[18,9] = 1 # letzte Zeile steht für output Node und dieser erhält ja operation output_rnn
    return op_matrix

# test2 = generate_ops() # ergibt 11x10 matrix

# generate_num = 10000 
# generate_num = 5 
def generate_archs(generate_num):
    archs = []
    archs_hash = []
    cnt = 0
    # es werden zuerst random adj und ops matritzen erzeugt und diese
    # werden dann einfach als dictionary eingespeichert und jedes dictionary bildet ein element einer liste archs
    while cnt < generate_num:
        adj_cnn = generate_adj_cnn()
        ops_cnn = generate_ops_cnn()
        
        adj_rhn = generate_adj_rhn()
        ops_rhn = generate_ops_rhn()
        
        if is_valid(adj_cnn, ops_cnn): # dieser Teil macht meiner meinung nach keinen Sinn
            arch = {"adjacency_matrix_cnn":adj_cnn, "operations_cnn":ops_cnn, "adjacency_matrix_rhn":adj_rhn, "operations_rhn":ops_rhn}
            arch_hash = str(hash(str(arch)))
            if arch_hash not in archs_hash:
                archs.append(arch)
                archs_hash.append(arch_hash)
                cnt += 1
    return archs

def is_valid(adj, op, step=4):
    for i in range(step):
        # i=0
        if (adj[:, 2 * i + 2] == adj[:, 2 * i + 3]).all() and (op[2 * i + 2, :] == op[2 * i + 3, :]).all(): # Inputs von [2,3] müssen nicht gleich sein / Spalten von adj (was aber immer True sein wird)
        # und 2te Bedinung ist meistens False, weil operationen gleich sein müssen
            return 0
    return 1

if __name__ =='__main__':
    t1 = time.time()
    arch = generate_archs(10000)
