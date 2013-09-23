import numpy as np

from .. covers.helpers import R_to_socp
from .. import prox

def make_SC_split(cover_func):
    """ Creates a simple consensus split using the cover function in
        `cover_func`. Requires that `cover_func` implement the prototype:
        
            list_of_equations, num_cols = cover_func(socp_data, num_partitions)
    """
    def split(socp_data, N):
        """Simple consensus splitting.
        Takes in a full socp description from QCML and returns a
        list of simple consensus prox operators
        """
        rho = 1
        c = socp_data['c']

        R_list, n = cover_func(socp_data, N)
        prox_list = []

        for R_data in R_list:
            local_socp = R_to_socp(R_data)
            local_socp['c'] = c/N
            p = prox.Prox(local_socp, rho)
            prox_list.append(p)

        return prox_list, N
    return split