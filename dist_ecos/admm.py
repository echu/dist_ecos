import numpy as np
from . split import SC_split
from . rformat.helpers import show_spy


class ADMM(object):
    settings = {
        'max_iters': 1000,
        'show_spy': False
    }

    def __init__(self, num_proxes=1, split_method=SC_split):
        self.split = split_method
        self.num_proxes = num_proxes

    def solve(self, socp_data):
        """ Solves the problem using ADMM
        """
        #import ipdb; ipdb.set_trace();
        n = socp_data['c'].shape[0]
        z = np.zeros((n))
        proxes = self.split(socp_data, self.num_proxes)

        if ADMM.settings['show_spy']:
            for p in proxes:
                show_spy(p.socp_vars)

        errs = []
        for j in xrange(ADMM.settings['max_iters']):
            print 'iter %d' % j

            total = np.zeros((n))
            count = np.zeros((n))

            for p in proxes:
                x, info = p.xupdate(z[p.global_index])
                total[p.global_index] += x
                count[p.global_index] += 1

            z_old = z
            z = total/count

            errs.append(np.linalg.norm(np.array(z-z_old)))

        return errs
