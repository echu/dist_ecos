import numpy as np

from .. covers.helpers import R_to_reduced_socp
from .. import prox


def make_GC_split(cover_func):
    """ Creates a general consensus split using the cover function in
        `cover_func`. Requires that `cover_func` implement the prototype:
        
            list_of_equations, num_cols = cover_func(socp_data, num_partitions)
    """
    def split(socp_data, N):
        """General consensus splitting
        Takes in a full socp description from QCML and returns a
        list of general consensus prox operators, and the weights
        for the global variable average"""
        rho = 1
        c = socp_data['c']

        R_list, n = cover_func(socp_data, N)
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
    return split