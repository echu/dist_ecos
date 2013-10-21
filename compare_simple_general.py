#!/usr/bin/env python
"""Compare simple consensus vs general consensus on the same problem"""
import dist_ecos.split as split
from dist_ecos import admm
from dist_ecos.rformat.helpers import show_spy

#'global' problem
import dist_ecos.problems.svc as gp

print gp.socp_vars
show_spy(gp.socp_vars)

runs = 1000

admm.settings['num_proxes'] = 6
admm.settings['max_iters'] = runs
admm.settings['show_spy'] = False

# tests[type] returns tuple with split method and whether to reset
tests = {
#    'simple':               (split.SC_split, False),
#    'simple with metis':    (split.SC_metis_split, False),
#    'simple with random':   (split.SC_random_split, False),
    'general':              (split.GC_split, False),
#    'general with metis':   (split.GC_metis_split, False),
#    'general with random':  (split.GC_random_split, False),
#    'simple int':           (split.SC_intersect, True),
#    'simple int metis':     (split.SC_metis_intersect, True),
#    'simple int random':    (split.SC_random_intersect, True),
#    'general int':          (split.GC_intersect, True),
#    'general int metis':    (split.GC_metis_intersect, True),
#    'general int random':   (split.GC_random_intersect, True),
#    'simple with laplacian': (split.SC_laplacian_split, False),
    'general with laplacian': (split.GC_laplacian_split, False)
}
results = {}

for label, test_params in tests.iteritems():
    print "method: ", label
    split_method, with_reset = test_params
    admm.settings['split_method'] = split_method
    results[label] = admm.solve(gp.socp_vars, with_reset)

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
    pylab.subplot(2, 1, 1)
    line = pylab.semilogy(range(runs), res_pri, style, color=color)
    lines.append(line[0])
    pylab.ylabel('primal residual')

    pylab.subplot(2, 1, 2)
    pylab.semilogy(range(runs), res_dual, style, color=color)
    pylab.xlabel('iteration')
    pylab.ylabel('dual residual')

pylab.figlegend(lines, tests.keys(), loc='upper center', shadow=True,
                fancybox=True, ncol=3)
pylab.show()
