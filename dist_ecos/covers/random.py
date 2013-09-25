from helpers import socp_to_R
import numpy as np

def start_row(n, k, i):
    '''start of ith partition of list of length n into k partitions'''
    d = n//k
    r = n % k
    return d*i + min(i, r)

def cover(socp_data, N):
    """stacks the socp data and partitions it into N
    local dicts describing constraints R <= s"""
    R, s, cone_array = socp_to_R(socp_data)

    local_list = []

    m, n = R.shape
    random_partition = np.random.randint(0,N,m)

    list_of_lists = [[] for i in xrange(N)]
    for i, group in enumerate(random_partition):
        list_of_lists[group].append(i)
    
    for i in xrange(N):
        local_R_data = {}
        local_R_data['R'] = R[list_of_lists[i], :]
        local_R_data['s'] = s[list_of_lists[i]]
        local_R_data['cone_array'] = np.array(cone_array)[list_of_lists[i]]
        
        local_list.append(local_R_data)


    return local_list, n