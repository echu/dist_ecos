import numpy as np
import itertools

from . covers import metis, naive, random, graclus, mondriaan, laplacian
from . consensus import general, simple

from . proxes.prox_list import ProxList
from . proxes.prox import Prox

split = {
    'naive':        naive.cover,
    'random':       random.cover,
    'graclus':      graclus.cover,
    'metis':        metis.cover,
    'mondriaan':    mondriaan.cover,
    'laplacian':    laplacian.cover
}

deal = {
    'simple':       simple.deal,
    'general':      general.deal,
}

def partition(socp_data, N, part):
    if socp_data['A']:
        p = socp_data['A'].shape[0]
    else:
        p = 0
        
    cone_array = np.hstack( (np.arange(p + socp_data['dims']['l'] + 1, dtype=np.int),
                            p + socp_data['dims']['l'] + np.cumsum(socp_data['dims']['q'], dtype=np.int)))
    # should have len(cone_array) == p + m + 1
    
    # this step performs the partition
    A_list_of_lists = [[] for i in xrange(N)]
    G_list_of_lists = [[] for i in xrange(N)]
    linear_cones = [0 for i in xrange(N)]
    soc_cones = [[] for i in xrange(N)]
    for i, group in enumerate(part):
        if i < p:
            A_list_of_lists[group].append(i)
        else:
            ind = i - p
            start = cone_array[ind]
            end = cone_array[ind+1]
            G_list_of_lists[group].extend(range(start,end))
            if end-start == 1:
                linear_cones[group] += 1
            else:
                soc_cones[group].append(int(end - start))
                
    return A_list_of_lists, G_list_of_lists, linear_cones, soc_cones



def split_problem(socp_data, user_options):    
    n = socp_data['c'].shape[0]
    N = user_options['N']

    split_func = split[user_options['split']]
    deal_func = deal[user_options['consensus']]

    cover = split_func(socp_data, N)
    cover_info = partition(socp_data, N, cover)
    socp_datas, indices = deal_func(socp_data, N, *cover_info)
    
    # count subsystems
    count = np.zeros((n))
    for index in indices:
        if index is not None:
            count[index] += 1
        else:
            count += 1
    
    # print diagnostic (average coupling between subsystems)
    total_shared_vars = np.zeros((N))
    for i, index in enumerate(indices):
        variables = count[index]
        shared_vars = variables[variables > 1]
        total_shared_vars[i] = len(shared_vars)
    print "average coupling:", np.mean(total_shared_vars)
    
    proxes = []
    for data, index in itertools.izip(socp_datas, indices):
        data['c'] = data['c'] / count
        if index is not None:
            # select out elements
            data['c'] = data['c'][index]
        proxes.append( Prox(data, count, global_index = index, **user_options) )
    
    return ProxList(n, proxes, **user_options)