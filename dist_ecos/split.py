""" Functions for partitioning. Should make this a class or interface.
"""
import scipy.sparse as sp
import numpy as np
import prox

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


def split_socp_to_R(socp_data, N):
    """stacks the socp data and partitions it into N
    local dicts describing constraints R <= s"""
    R, s, cone_array = socp_to_R(socp_data)

    local_list = []

    m, n = R.shape

    for i in range(N):
        local_R_data = {}
        start = start_row(m, N, i)
        stop = start_row(m, N, i+1)

        local_R_data['R'] = R[start:stop, :]
        local_R_data['s'] = s[start:stop]
        local_R_data['cone_array'] = cone_array[start:stop]

        local_list.append(local_R_data)

    return local_list, n

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

def split_socp_to_R_using_metis(socp_data, N):
    """stacks the socp data and partitions it into N
    local dicts describing constraints R <= s"""
    import pymetis as pm
    import networkx as nx
    import pylab
        
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
    for i in range(len(s)):
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


def GC_split(socp_data, N):
    """General consensus splitting
    Takes in a full socp description from QCML and returns a
    list of general consensus prox operators, and the weights
    for the global variable average"""
    rho = 1
    c = socp_data['c']

    R_list, n = split_socp_to_R(socp_data, N)
    local_socp_list = []
    prox_list = []

    c_count = np.zeros((n))

    #convert form R-form to socp form
    #also, count the global variables, to normalize 'c' correctly later
    for R_data in R_list:
        local_socp, global_index = R_to_reduced_socp(R_data)
        c_count[global_index] += 1
        local_socp_list.append((local_socp, global_index))

    #form the local socp prox objects, with correctly normalized c
    for local_socp, global_index in local_socp_list:
        local_socp['c'] = c[global_index]/c_count[global_index]
        p = prox.GCProx(local_socp, global_index, rho)
        prox_list.append(p)

    return prox_list, c_count

def GC_metis_split(socp_data, N):
    """General consensus splitting
    Takes in a full socp description from QCML and returns a
    list of general consensus prox operators, and the weights
    for the global variable average. It uses metis for the
    partitioning."""
    rho = 1
    c = socp_data['c']

    R_list, n = split_socp_to_R_using_metis(socp_data, N)
    local_socp_list = []
    prox_list = []

    c_count = np.zeros((n))

    #convert form R-form to socp form
    #also, count the global variables, to normalize 'c' correctly later
    for R_data in R_list:
        local_socp, global_index = R_to_reduced_socp(R_data)
        c_count[global_index] += 1
        local_socp_list.append((local_socp, global_index))

    #form the local socp prox objects, with correctly normalized c
    for local_socp, global_index in local_socp_list:
        local_socp['c'] = c[global_index]/c_count[global_index]
        p = prox.GCProx(local_socp, global_index, rho)
        prox_list.append(p)

    return prox_list, c_count


def SC_split(socp_data, N):
    """Simple consensus splitting.
    Takes in a full socp description from QCML and returns a
    list of simple consensus prox operators
    """
    rho = 1
    c = socp_data['c']

    R_list, n = split_socp_to_R(socp_data, N)
    prox_list = []

    for R_data in R_list:
        local_socp = R_to_socp(R_data)
        local_socp['c'] = c/N
        p = prox.Prox(local_socp, rho)
        prox_list.append(p)

    return prox_list

def SC_metis_split(socp_data, N):
    """Simple consensus splitting using PyMetis partitioner.
    Takes in a full socp description from QCML and returns a
    list of simple consensus prox operators
    """
    rho = 1
    c = socp_data['c']

    R_list, n = split_socp_to_R_using_metis(socp_data, N)
    prox_list = []

    for R_data in R_list:
        local_socp = R_to_socp(R_data)
        local_socp['c'] = c/N
        p = prox.Prox(local_socp, rho)
        prox_list.append(p)

    return prox_list

def form_laplacian_ordering(socp_data):
    pass


def form_shuffle(socp_data):
    pass


def start_row(n, k, i):
    '''start of ith partition of list of length n into k partitions'''
    d = n//k
    r = n % k
    return d*i + min(i, r)


if __name__ == '__main__':
    #import problems.basis_pursuit as gp
    import problems.svc as gp

    print 'gp socpvars is ', gp.socp_vars
    show_spy(gp.socp_vars)

    proxes, c_count = GC_split(gp.socp_vars, 5)
    for p in proxes:
        show_spy(p.socp_vars)
        print p.global_index
    print 'c_count is ', c_count
