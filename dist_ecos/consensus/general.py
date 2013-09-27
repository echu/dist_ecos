from .. rformat.to_socp import R_to_reduced_socp
from .. rformat.standard import socp_to_R
from .. import prox
import numpy as np


def make_GC_split(cover_func):
    """ Creates a general consensus split using the cover function in
        `cover_func`. Requires that `cover_func` implement the prototype:

            list_of_equations = cover_func(R, s, cone_array, num_partitions)
    """
    def split(socp_data, N):
        """ General consensus splitting
            Takes in a full socp description from QCML and returns a
            list of general consensus prox operators
        """
        rho = 1
        c = socp_data['c']
        n = c.shape[0]

        R, s, cone_array = socp_to_R(socp_data)

        R_list = cover_func(R, s, cone_array, N)
        local_socp_list = []
        prox_list = []

        count = np.zeros((n))

        #convert form R-form to socp form
        #also, count the global variables, to normalize 'c' correctly later
        for R_data in R_list:
            local_socp, global_index = R_to_reduced_socp(R_data)
            count[global_index] += 1
            local_socp_list.append((local_socp, global_index))

        #form the local socp prox objects, with correctly normalized c
        for local_socp, global_index in local_socp_list:
            local_socp['c'] = c[global_index]/count[global_index]
            p = prox.GCProx(local_socp, global_index, rho)
            prox_list.append(p)

        #let's get rid of c_count and just let admm figure it out based on
        #how many proxes touch each variable element
        return prox_list
    return split
