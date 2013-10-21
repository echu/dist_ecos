import numpy as np
import scipy.sparse as sp
from . utils import collapse_cones
''' Compute matrix rows similarity and produce an ordering using the
    second eigenvalue of the Laplacian matrix.

    The two most interesting similarities to use seem to be the Jaccard
    similarity and the cosine similarity.
'''


def abs_sim(G):
    G = np.abs(G)
    S = G.dot(G.T)
    return S


def sparse_sim(G):
    '''similarity based only on sparsity pattern'''
    ind = np.nonzero(G)
    G = G.copy()
    G[ind] = 1.0
    S = G.dot(G.T)
    return S
    #let's also try normalizing this


def jaccard_sim(G):
    '''compute jaccard similarity between vector sparisty patterns'''
    ind = np.nonzero(G)
    G = G.copy()
    G[ind] = 1.0
    AintB = G.dot(G.T)
    lens = np.sum(G, 1)
    AunionB = lens[:, np.newaxis] + lens[np.newaxis, :] - AintB
    return AintB/AunionB


def abs_sim2(G):
    S = G.dot(G.T)
    S = np.abs(S)
    d = 1.0/(np.sqrt(np.diag(S))+np.sqrt(np.spacing(1)))
    S = d[np.newaxis, :]*S
    S = d[:, np.newaxis]*S
    return S


def cos_sim(G):
    '''cosine similarity matrix'''
    S = G.dot(G.T)
    #want the result between -1 and 1
    d = 1.0/(np.sqrt(np.diag(S))+np.sqrt(np.spacing(1)))
    S = d[np.newaxis, :]*S
    S = d[:, np.newaxis]*S
    S = 1.0 - np.arccos(S)/np.pi
    return S


def sort_by_sim(S, reg=1):
    """higher reg encourages order to stay unchanged"""
    m, n = S.shape
    neighbor_sim = reg*np.min(S[np.nonzero(S)])

    #we add off-diagonal terms to make sure we have a unique second eigenvector
    #the off-diagonal terms encourage the rows to stay in their current order
    S2 = S + np.diag(neighbor_sim*np.ones(n-1), 1) +\
        np.diag(neighbor_sim*np.ones(n-1), -1)

    d2 = np.sum(S2, 1)
    L2 = np.diag(d2) - S2

    import numpy.linalg as la
    w, v = la.eigh(L2)
    sort_indx = np.argsort(w)

    order = np.argsort(v[:, sort_indx[1]])
    #plot(v[order,sort_indx[1]])
    return order

def start_row(n, k, i):
    '''start of ith partition of list of length n into k partitions'''
    d = n//k
    r = n % k
    return d*i + min(i, r)

def cover_order(order, m, N):
    local_list = []

    part = np.zeros((m), dtype=np.int)
    for i in xrange(N):
        local_R_data = {}
        start = start_row(m, N, i)
        stop = start_row(m, N, i+1)

        part[ order[start:stop] ] = i

    return part


def cover(socp_data, N):
    #S = jaccard_sim(R.toarray())
    if socp_data['A'] is not None:
        R = sp.vstack((socp_data['A'], collapse_cones(socp_data)))
    else:
        R = collapse_cones(socp_data)
    S = cos_sim(R.toarray())
    order = sort_by_sim(S)
    
    cone_lengths = socp_data['dims']['l'] + len(socp_data['dims']['q'])
    
    return cover_order(order, cone_lengths, N)
