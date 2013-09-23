import pymetis as pm
import networkx as nx
import pylab
import numpy as np
import scipy.sparse as sp

from helpers import socp_to_R
    
def form_laplacian(A):
    """Form the Laplacian of a sparse, rectangular matrix.
    """
    A = A.tocoo()
    m, n = A.shape
    
    ii = np.hstack((A.row + n, A.col))
    jj = np.hstack((A.col, A.row + n))
    vv = np.hstack((A.data, A.data))
    
    symA = sp.coo_matrix((vv, (ii,jj)), (m+n,m+n))
    return sp.csgraph.laplacian(symA)

def cover(socp_data, N):
    """stacks the socp data and partitions it into N
    local dicts describing constraints R <= s"""

    R, s, cone_array = socp_to_R(socp_data)
    # form the Laplacian and use pymetis to partition
    L = form_laplacian(R)
    graph = nx.from_scipy_sparse_matrix(L)
    cuts, part_vert = pm.part_graph(N, graph)
    
    local_list = []
    m, n = R.shape
    
    list_of_lists = [[] for i in xrange(N)]
    for i, group in enumerate(part_vert[n:]):
        list_of_lists[group].append(i)
    
    
    # # for plotting...
    # import pylab
    # pylab.figure(1)
    # pylab.subplot(211)
    # 
    # pylab.spy(R, marker='.')
    # 
    # pylab.subplot(212)
    # 
    # color = "rgbcmyk"
    # 
    # for i in xrange(N):
    #     H = R[list_of_lists[i],:].tocoo()
    #     row = np.array(list_of_lists[i])[H.row]
    #     col = H.col
    #     values = H.data
    #     to_show = sp.coo_matrix((values, (row,col)), (m,n))
    #     pylab.spy(to_show, marker='.', color=color[i])
    # 
    # 
    # 
    # pylab.show()
    
    for i in xrange(N):
        local_R_data = {}
        local_R_data['R'] = R[list_of_lists[i], :]
        local_R_data['s'] = s[list_of_lists[i]]
        local_R_data['cone_array'] = np.array(cone_array)[list_of_lists[i]]
        
        local_list.append(local_R_data)
    
    return local_list, n