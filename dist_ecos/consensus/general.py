from .. rformat.to_socp import R_to_reduced_socp
#from .. rformat.standard import socp_to_R
from .. import prox
import numpy as np


def make_GC_split(cover_func, conversion_func):
    """ Creates a general consensus split using the cover function in
        `cover_func` and the SOCP conversion function in `conversion_func`.
         Requires that `cover_func` implement the prototype:

            list_of_equations = cover_func(R, s, cone_array, num_partitions)
        
        Requires that `conversion_func` implement the prototype:
        
            R, s, cone_array, is_intersect = conversion_func(socp_data)
        
        where socp_data is a dictionary containing
            {'c': objective vector,
             'A': equality constraint matrix
             'b': equality constraint vector
             'G': cone inequality matrix
             'h': cone inequality vector
             'dims': list of cones}
    """
    def split(socp_data, N):
        """ General consensus splitting
            Takes in a full socp description from QCML and returns a
            list of general consensus prox operators
        """
        rho = 1

        R, s, cone_array, is_intersect = conversion_func(socp_data)
        
        c = socp_data['c']
        n = R.shape[1]

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
            if is_intersect:
                local_socp['c'] = None
            else:
                local_socp['c'] = c[global_index]/count[global_index]
            p = prox.GCProx(local_socp, global_index, rho)
            prox_list.append(p)

        #let's get rid of c_count and just let admm figure it out based on
        #how many proxes touch each variable element
        #
        #echu: now returns the length of the global variable
        return prox_list, n
    return split
