
import numpy as np
import scipy.sparse as sp

def squish(A,G):
    if A is not None:
        A = A.tocoo()
        p, n = A.shape
        Acol = A.col
    else:
        p = 0
        Acol = np.empty(0)
    
    if G is not None:
        G = G.tocoo()
        m, n = G.shape
        Gcol = G.col
    else:
        m = 0
        Gcol = np.empty(0)
    
    vars_touched = np.hstack((Acol, Gcol))
    global_index, cols = np.unique(vars_touched, return_inverse = True)
    
    if A is not None:
        A = sp.coo_matrix((A.data, (A.row, cols)), shape=(p, len(global_index)))
    if G is not None:
        G = sp.coo_matrix((G.data, (G.row, cols)), shape=(m, len(global_index)))
    
    return A, G, np.array(global_index, dtype=np.int)
    

def deal(socp_data, N, A_ind, G_ind, linear, soc):
    socp_datas = []
    indices = []
    for i in xrange(N):
        local_socp_data = {'c': socp_data['c']}
        # A_ind[i] indexes into A, G_ind[i] indexes into G
        
        # if equality constraints exist and the group has elements
        if socp_data['A'] is not None and A_ind[i]:
            local_socp_data['A'] = socp_data['A'][A_ind[i], :]
            local_socp_data['b'] = socp_data['b'][A_ind[i]]
        else:
            local_socp_data['A'] = None
            local_socp_data['b'] = None
        
        if G_ind[i]:
            local_socp_data['G'] = socp_data['G'][G_ind[i], :]
            local_socp_data['h'] = socp_data['h'][G_ind[i]]
        else:
            local_socp_data['G'] = None
            local_socp_data['h'] = None
        A, G = local_socp_data['A'], local_socp_data['G']
        A, G, index = squish(A, G)
        local_socp_data['A'], local_socp_data['G'] = A, G
        
        local_socp_data['dims'] = {'l': linear[i], 'q': soc[i], 's': []}
        indices.append(index)     # global index
        socp_datas.append(local_socp_data)
    return socp_datas, indices
