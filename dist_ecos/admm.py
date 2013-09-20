import numpy as np
from . split import SC_split
from . covers import helpers

class ADMM(object):
    settings = {
        'max_iters': 1000,
        'show_spy': False
    }
    
    def __init__(self, num_proxes = 1, split_method = SC_split):
        self.split = split_method
        self.num_proxes = num_proxes
    
    def solve(self, prob):
        """ Solves the problem using ADMM
        """
        n = prob.n
        z = np.zeros((n))
        proxes, N = self.split(prob.socp_vars, self.num_proxes)
        
        if ADMM.settings['show_spy']:
            for p in proxes:
                helpers.show_spy(p.socp_vars)
        
        errs = []
        for j in xrange(ADMM.settings['max_iters']):
            print 'iter %d' % j

            total = np.zeros((n))

            for p in proxes:
                x, info = p.xupdate(z[p.global_index])
                total[p.global_index] += x

            z_old = z
            z = total/N

            errs.append(np.linalg.norm(np.array(z-prob.socp_sol)))
        
        return errs
    


