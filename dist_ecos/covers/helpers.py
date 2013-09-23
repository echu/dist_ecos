""" This module contains all the functions needed for splitting.
"""
import scipy.sparse as sp
import numpy as np

def show_spy(socp_vars):
    '''Show the sparsity pattern of A and G in
    the socp_vars dictionary
    '''
    import pylab

    pylab.figure(1)
    pylab.subplot(211)
    #print 'A is', socp_vars['A']
    if socp_vars['A'] is not None:
        pylab.spy(socp_vars['A'], marker='.')
    pylab.xlabel('A')

    #print 'G is', socp_vars['G']
    pylab.subplot(212)
    pylab.spy(socp_vars['G'], marker='.')
    pylab.xlabel('G')

    pylab.show()


def socp_to_R(socp_data):
    """Stack the equality and inequality constraints.
    Return stacked structure, and cone_array to describe what cone
    is associated with each row.
    """
    mat_list = []
    vec_list = []
    z, l = 0, 0

    if socp_data['A'] is not None:
        mat_list.append(socp_data['A'])
        vec_list.append(socp_data['b'])
        z = socp_data['A'].shape[0]
    if socp_data['G'] is not None:
        mat_list.append(socp_data['G'])
        vec_list.append(socp_data['h'])
        l = socp_data['G'].shape[0]

    R = sp.vstack(mat_list, format='csr')
    s = np.concatenate(vec_list)

    cone_array = ['z']*z + ['l']*l

    return R, s, cone_array


def R_to_socp(R_data):
    """Converts from R-format to scop format, producing a dictionary
    with A, b, G, h, and dims. 'c' should be added later. We don't
    add it now only because, in the case of general consensus,
    we don't know how to scale c.
    """
    R = R_data['R']
    s = R_data['s']
    cone_array = R_data['cone_array']
    R = R.tocsr()

    m, n = R.shape

    #convert back to A, G form
    Aind = []
    Gind = []
    for i in xrange(len(s)):
        if cone_array[i] == 'z':
            Aind.append(i)
        elif cone_array[i] == 'l':
            Gind.append(i)
        else:
            raise Exception('Invalid cone_array.')

    if len(Aind) > 0:
        A = R[Aind, :]
        b = s[Aind]
    else:
        A = None
        b = None

    if len(Gind) > 0:
        G = R[Gind, :]
        h = s[Gind]
    else:
        G = None
        h = None

    dims = {'l': len(Gind), 'q': [], 's': []}

    #we can't yet add c, because we don't know the right scaling
    local_socp_data = {'A': A, 'b': b, 'G': G, 'h': h,
                       'dims': dims}

    return local_socp_data

def R_to_reduced_socp(R_data):
    '''converts a sparse matrix to a compressed form by removing
    the colums with only zero entries.
    returns the matrix and a mapping of the new local variables to
    the original global variables they correspond to'''
    R = R_data['R']
    s = R_data['s']

    #reduce to the nonzero columns
    R = R.tocoo()
    global_index, cols = np.unique(R.col, return_inverse=True)

    R = sp.coo_matrix((R.data, (R.row, cols)),
                      shape=(R.shape[0], len(global_index)))
    R = R.tocsr()

    R_data['R'] = R
    R_data['s'] = s

    #transform back to socp form
    local_socp_data = R_to_socp(R_data)

    return local_socp_data, global_index

