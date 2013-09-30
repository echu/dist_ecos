#!/usr/bin/env python
"""Compare simple consensus vs general consensus on the same problem"""
import dist_ecos.split as split
from dist_ecos import ADMM
from dist_ecos.rformat.helpers import show_spy

#'global' problem
import dist_ecos.problems.svc as gp

print gp.socp_vars
show_spy(gp.socp_vars)

runs = 1000

ADMM.settings['num_proxes'] = 4
ADMM.settings['max_iters'] = runs

tests = {
    'simple':               split.SC_split,
    'simple with metis':    split.SC_metis_split,
    'simple with random':   split.SC_random_split,
    'general':              split.GC_split,
    'general with metis':   split.GC_metis_split,
    'general with random':  split.GC_random_split,
    'simple int':           split.SC_intersect,
    'simple int metis':     split.SC_metis_intersect,
    'simple int random':    split.SC_random_intersect,
    'general int':          split.GC_intersect,
    'general int metis':    split.GC_metis_intersect,
    'general int random':   split.GC_random_intersect
}
results = {}

for label, split_method in tests.iteritems():
    ADMM.settings['split_method'] = split_method
    results[label] = ADMM.solve(gp.socp_vars)

import pylab
lines = []
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
styles = ['_', '-', '--', ':']

count = 0
for label in tests.keys():
    count += 1
    color = colors[count % len(colors)]
    style = styles[count % len(styles)]
    res_pri = results[label]['res_pri']
    res_dual = results[label]['res_dual']
    #errs = results[label]['errs']
    pylab.subplot(2,1,1)
    line = pylab.semilogy(range(runs), res_pri, style, color=color)
    lines.append(line[0])
    pylab.ylabel('primal residual')
    
    pylab.subplot(2,1,2)
    pylab.semilogy(range(runs), res_dual, style, color=color)
    pylab.xlabel('iteration')
    pylab.ylabel('dual residual')

pylab.figlegend(lines, tests.keys(), loc='upper center', shadow=True, fancybox=True, ncol=3)
pylab.show()



