import numpy as np
import scipy.sparse as sp


def collapse_cones(socp_data):
    """ Take a G matrix and collapse the rows so that the number of rows
        is equal to the number of cones.
    """
    dims, G = socp_data['dims'], socp_data['G']
    if dims['q']:
        A = G[:dims['l'],:]
        
        start = dims['l']
        for n in dims['q']:
            end = start + n
            A = sp.vstack(
                (A,abs(G[start:end,:]).sum(0))
            )
        return A
    else:
        return G
        
def form_laplacian(socp_data):
    """ Form the Laplacian of a sparse, rectangular matrix.
        Create the bipartite graph
        [ 0 A^T G^T
          A  0   0
          G  0   0 ]
    """
    A = socp_data['A']
    # collapse any SOC cones in G
    G = collapse_cones(socp_data)
    if A is not None:
        A = A.tocoo()
    G = G.tocoo()
    
    if A is not None:
        p, n = A.shape
    else:
        p = 0
    m, n = G.shape
    
    if A is not None:
        ii = np.hstack((A.row + n, A.col, G.row + n + p, G.col))
        jj = np.hstack((A.col, A.row + n, G.col, G.row + n + p))
        vv = np.hstack((A.data, A.data, G.data, G.data))
    else:
        ii = np.hstack((G.row + n, G.col))
        jj = np.hstack((G.col, G.row + n))
        vv = np.hstack((G.data, G.data))
    
    symA = sp.coo_matrix((vv, (ii, jj)), (m+n+p, m+n+p))
    
    # this forms [A;G]*[A;G]^T
    # if A is not None:
    #     ii = np.hstack((A.row, G.row + p))
    #     jj = np.hstack((A.col, G.col))
    #     vv = np.hstack((A.data, G.data))
    #     AG = sp.coo_matrix((vv, (ii, jj)), (m+p, m+p))
    # else:
    #     AG = G
    # 
    # symA = AG*AG.T
    return sp.csgraph.laplacian(symA)
