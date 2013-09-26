""" Takes socp_data/ECOS format and transforms it into R format to be split
    across processors.
"""
import numpy as np
import scipy.sparse as sp


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
