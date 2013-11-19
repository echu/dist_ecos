""" Takes socp_data and transforms it into an R-format system without an
    objective so that the problem can be solved as a pure convex intersection
    problem.
"""
import numpy as np
import scipy.sparse as sp
from . import standard


def convert(socp_data):
    """ Convert socp_data to a pure convex intersection format. The form is
        stored as R <= s where the cone types are described in cone_array.
    """
    R, s, cone_array, is_intersect = standard.convert(socp_data)

    m, n = R.shape

    s_list = [[0], s, -1 * socp_data['c']]
    cone_list = [['z'], cone_array, ['z'] * n]
    mat_list = [[socp_data['c'], s], [R, None], [None, R.T]]

    ind = []
    for i, elem in enumerate(cone_array):
        if elem == 'l':
            ind.append(i)

    # if there are LP constraints
    if len(ind) > 0:
        cone_lp = sp.coo_matrix(
            (-1 * np.ones(len(ind)), (range(len(ind)), ind)),
            shape=(len(ind), m))
        mat_list.append([None, cone_lp])
        s_list.append(np.zeros(len(ind)))
        cone_list.append(['l'] * len(ind))

    R = sp.bmat(mat_list, format='csr')
    s = np.concatenate(s_list)
    cone_array = [cone for sublist in cone_list for cone in sublist]

    return R, s, cone_array, True

