import numpy as np
from . split import SC_split
from . rformat.helpers import show_spy


settings = {
    'num_proxes': 1,
    'split_method': SC_split,
    'max_iters': 1000,
    'show_spy': False
}


def solve(socp_data):
    """ Solves the problem using ADMM
    """
    import math
    #import ipdb; ipdb.set_trace();
    proxes, n = settings['split_method'](socp_data, settings['num_proxes'])
    
    # TODO: need a better way to keep track of "global" N
    #n = proxes[0].n
    z = np.zeros((n))

    if settings['show_spy']:
        for p in proxes:
            show_spy(p.socp_vars)

    errs = []
    res_pri = []
    res_dual = []
    for j in xrange(settings['max_iters']):
        print 'iter %d' % j

        z_total = np.zeros((n))
        z_count = np.zeros((n))

        pri_total = 0.0
        dual_total = 0.0
        for p in proxes:
            x, info = p.xupdate(z[p.global_index])
            z_total[p.global_index] += x
            z_count[p.global_index] += 1

            pri_total += info['primal']
            dual_total += info['dual']

        z_old = z
        z = z_total/z_count

        #should remove this, just leaving it here for now because
        #it should match the dual residual in the simple consensus case
        errs.append(math.sqrt(len(proxes))*np.linalg.norm(np.array(z-z_old)))

        res_pri.append(math.sqrt(pri_total))
        res_dual.append(math.sqrt(dual_total))

        result = {'errs': errs, 'res_pri': res_pri, 'res_dual': res_dual}

    return result
