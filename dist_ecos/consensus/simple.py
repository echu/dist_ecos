from .. rformat.to_socp import R_to_socp
#from .. rformat.standard import socp_to_R
from .. import prox


def make_SC_split(cover_func, conversion_func):
    """ Creates a simple consensus split using the cover function in
        `cover_func` and the SOCP conversion function in `conversion_func`.
         Requires that `cover_func` implement the prototype:

            list_of_equations = cover_func(R, s, cone_array, num_partitions)
        
        Requires that `conversion_func` implement the prototype:
        
            R, s, cone_array = conversion_func(socp_data)
        
        where socp_data is a dictionary containing
            {'c': objective vector,
             'A': equality constraint matrix
             'b': equality constraint vector
             'G': cone inequality matrix
             'h': cone inequality vector
             'dims': list of cones}
    """
    def split(socp_data, N):
        """Simple consensus splitting.
        Takes in a full socp description from QCML and returns a
        list of simple consensus prox operators
        """
        rho = 1
        c = socp_data['c']

        R, s, cone_array, is_intersect = conversion_func(socp_data)

        R_list = cover_func(R, s, cone_array, N)
        prox_list = []

        for R_data in R_list:
            local_socp = R_to_socp(R_data)
            if is_intersect:
                local_socp['c'] = None
            else:
                local_socp['c'] = c/N
            p = prox.Prox(local_socp, rho)
            prox_list.append(p)

        return prox_list, R.shape[1]
    return split
