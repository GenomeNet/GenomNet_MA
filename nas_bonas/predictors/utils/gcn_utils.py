import numpy as np


# mx, ifAdj, maxsize = trained_arch_list[0]['adjacency_matrix'], True, 7



def add_global_node(mx, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if (ifAdj):
        mx = np.column_stack((mx, np.ones(mx.shape[0], dtype=np.float32))) # fügt einen 11x1 Vektor nur mit 1en dazu (als neue Spalte) -> jetzt 12te column von mx
        # was natürlich extrem sinn macht, weil unser global Node durch diese neue Spalte repräsentiert wird und er ja von allen anderen Nodes, einen Input erhält!!
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32))) # fügt einen 1x12 Vektor nur mit 0en dazu (als neue Zeile)
        # brauche ich ja, weil ich neue Spalte hinzugefügt habe und adj matritzen haben immer gleich viele zeilen wie spalten (quadratische matrix)
        np.fill_diagonal(mx, 1) # diagonale mit 1en ausfüllen: wrsl. so ne sache von diesem global node paper
        mx = mx.T # transponieren
    else:
        mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        mx[mx.shape[0] - 1][mx.shape[1] - 1] = 1
    return mx



# mx, ifAdj, maxsize = np.array(sample['adjacency_matrix']), True, 7
def padzero(mx, ifAdj, maxsize=7):
    if ifAdj:
        while mx.shape[0] < maxsize: # er fügt so häufig 0er Spalten und 0er Zeilen hinzu, bis die erforderliche Dimension von maxsize=7 erfüllt ist
            mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
            mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
    else:
        while mx.shape[0] < maxsize:
            mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
    return mx




def net_decoder(operations):
    operations = np.array(operations, dtype=np.int32)
    for i in range(len(operations)):
        if operations[i] == 2: #input
            operations[i]=3
        elif operations[i] == 3: #conv1
            operations[i]=0
        elif operations[i] == 4: #pool
            operations[i]=2
        elif operations[i] == 5: #conv3
            operations[i]=1
        elif operations[i] == 6: #output
            operations[i]=4
    one_hot = np.zeros((len(operations), 5))
    one_hot[np.arange(len(operations)), operations] = 1
    return one_hot

if __name__=="__main__":
    # a = np.array([[0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
    #    [0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0.],
    #    [0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1.],
    #    [0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1.],
    #    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
    #    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
    #    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    #    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    #    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    #    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    #    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    print("buya")
    a = np.array([[0., 0., 1., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],])
    print(a.shape)
    a = padzero(np.array(a), True, maxsize=11)
    print(a.shape)
    # add_global_node(padzero(np.array(a['adjacency_matrix']), True, maxsize=11), True)
