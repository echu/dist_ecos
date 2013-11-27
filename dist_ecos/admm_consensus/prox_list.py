'''
    converts socp data to a list of prox objects

    input: socp data and a list of the ways to partition rows
'''
import numpy as np
import scipy.sparse as sp
from itertools import izip
from prox import Prox


def form_prox_list(socp_data, cone_cover, rho=1, general=True):
    """
        cone cover is a list of lists. each list gives the cones
        ('rows' of system [A; G]) that correspond to a subsystem

        general is if we are doing general consensus
    """
    n = socp_data['c'].shape[0]
    #get the local socp row indices
    socp_infos = [socp_info(socp_data, rows) for rows in cone_cover]
    #form the local socp data matrices
    local_socp_datas = [form_local(socp_data, *info) for info in socp_infos]
    #general or simple consensus. changes data in-place
    if general:
        global_indices = [squish_socp(data) for data in local_socp_datas]
    else:
        global_indices = [range(n) for data in local_socp_datas]

    #count the number of subsystems each variable is in. properly
    #normalize c for each subsystem
    global_count = np.zeros(n)
    for ind in global_indices:
        global_count[ind] += 1
    for socp_data, ind in izip(local_socp_datas, global_indices):
        socp_data['c'] = socp_data['c']/global_count[ind]

    #now we make the list of proxes
    prox_list = []
    for socp_data, indx in izip(local_socp_datas, global_indices):
        prox_list.append(Prox(socp_data, global_count, global_index=indx, rho=rho))

    return prox_list


def socp_info(socp_data, cone_rows):
    """
        cone_rows is a list (not np.array) of the cones for this subsystem.
        these cones correspond to rows of the squished [A; G] system

        return which rows of A to select for this subsystem, and rows of the
        original (un-squished) G, along with cone dimension info

    """
    if socp_data['A'] is not None:
        p = socp_data['A'].shape[0]
    else:
        p = 0
    l = socp_data['dims']['l']
    q = socp_data['dims']['q']
    A_list = []
    G_list = []
    linear_cones = 0
    soc_cones = []

    print 'dims is ', socp_data['dims']
    print 'p is ', p
    print 'l is ', l
    print 'q is ', q

    for row in sorted(cone_rows):
        if row < p:
            A_list.append(row)
        elif row < p+l:
            G_list.append(row-p)
            linear_cones += 1
        else:
            print 'row is ', row

            soc_num = row - p - l
            print 'soc_num is ', soc_num
            start = sum(q[:soc_num]) + l
            stop = start + q[soc_num]
            G_list.extend(range(start, stop))
            soc_cones.append(q[soc_num])

    return A_list, G_list, linear_cones, soc_cones


def squish_socp(socp_data):
    """change socp data in place, but return the global index
    """
    A, G, global_index = squish(socp_data['A'], socp_data['G'])
    socp_data['A'] = A
    socp_data['G'] = G
    socp_data['c'] = socp_data['c'][global_index]
    return global_index


def squish(A, G):
    """ reduce a local A, G sparse system to just the
        variables with nonzero elements. return the mapping
        back to the global index
    """
    if A is not None:
        A = A.tocoo()
        p, n = A.shape
        Acol, Annz = A.col, A.nnz
    else:
        p = 0
        Acol, Annz = np.empty(0), 0

    if G is not None:
        G = G.tocoo()
        m, n = G.shape
        Gcol = G.col
    else:
        m = 0
        Gcol = np.empty(0)

    vars_touched = np.hstack((Acol, Gcol))
    global_index, cols = np.unique(vars_touched, return_inverse=True)

    if A is not None:
        A = sp.coo_matrix(
            (A.data, (A.row, cols[:Annz])), shape=(p, len(global_index)))
    if G is not None:
        G = sp.coo_matrix(
            (G.data, (G.row, cols[Annz:])), shape=(m, len(global_index)))

    return A, G, np.array(global_index, dtype=np.int)


def form_local(socp_data, A_ind, G_ind, linear, soc):
    """ take local subsystem index information and deal it out
        local socp data consisting of full numpy arrays.
        numpy arrays are reduced to just the variables that they touch
    """

    local_socp_data = {'c': socp_data['c']}
    # A_ind[i] indexes into A, G_ind[i] indexes into G

    # if equality constraints exist and the group has elements
    if socp_data['A'] is not None and A_ind:
        local_socp_data['A'] = socp_data['A'][A_ind, :]
        local_socp_data['b'] = socp_data['b'][A_ind]
    else:
        local_socp_data['A'] = None
        local_socp_data['b'] = None

    if G_ind:
        local_socp_data['G'] = socp_data['G'][G_ind, :]
        local_socp_data['h'] = socp_data['h'][G_ind]
    else:
        local_socp_data['G'] = None
        local_socp_data['h'] = None

    local_socp_data['dims'] = {'l': linear, 'q': soc, 's': []}

    return local_socp_data

if __name__ == '__main__':
    A = np.zeros((4, 5))
    dims = {'l': 6, 'q': [3, 2, 4, 5]}

    socp_data = {'A': A, 'dims': dims}

    cone_rows = range(14)

    print socp_info(socp_data, cone_rows)
