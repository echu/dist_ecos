""" Functions for partitioning. Should make this a class or interface.
"""










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





