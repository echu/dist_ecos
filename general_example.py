#!/usr/bin/env python
"""Run an example of general form consensus"""
import dist_ecos.split as split
from dist_ecos.rformat.helpers import show_spy
from dist_ecos import ADMM


#'global' problem
import dist_ecos.problems.svc as gp

print gp.socp_vars
show_spy(gp.socp_vars)

ADMM.settings['max_iters'] = 1000
ADMM.settings['show_spy'] = True
ADMM.settings['num_proxes'] = 5

#general consensus
ADMM.settings['split_method'] = split.GC_split
result = ADMM.solve(gp.socp_vars)

import pylab
res_pri = result['res_pri']
res_dual = result['res_dual']
errs = result['errs']

r = len(errs)
pylab.semilogy(range(r), errs, range(r), res_pri, range(r), res_dual)
pylab.legend(['errs', 'primal', 'dual'])
pylab.show()
