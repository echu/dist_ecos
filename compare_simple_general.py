#!/usr/bin/env python
"""Compare simple consensus vs general consensus on the same problem"""
import numpy as np
import dist_ecos.split as split
from dist_ecos.admm import ADMM
from dist_ecos.covers.helpers import show_spy



#'global' problem
import dist_ecos.problems.svc as gp

print gp.socp_vars
show_spy(gp.socp_vars)

num_proxes = 8
runs = 1000

tests = {
    'simple':               split.SC_split, 
    'simple with metis':    split.SC_metis_split,
    'simple with random':   split.SC_random_split,
    'general':              split.GC_split,
    'general with metis':   split.GC_metis_split,
    'general with random':  split.GC_random_split
}
results = {}

for label, split_method in tests.iteritems():
    solver = ADMM(num_proxes, split_method)
    results[label] = solver.solve(gp)

import pylab
for label in tests.keys():
    pylab.semilogy(range(runs), results[label])
pylab.legend(tests.keys())
pylab.show()
