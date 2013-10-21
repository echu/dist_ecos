#!/usr/bin/env python

"""Compare simple consensus vs general consensus on the same problem"""
import dist_ecos.split as split
from dist_ecos import consensus_conic_opt
from dist_ecos.rformat.helpers import show_spy

#'global' problem
import dist_ecos.problems.svc as gp

print gp.socp_vars
show_spy(gp.socp_vars)

runs = 100
N = 16

# tests[type] returns tuple with split method
tests = {
    'naive':        ('naive', 'simple'),
    'general':      ('naive', 'general'),
    'graclus':      ('graclus', 'general'),
    'mondriaan':    ('mondriaan', 'general'),
    'random':       ('random', 'general'),
    'metis':        ('metis', 'general')
}
results = {}


for label, test_params in tests.iteritems():
    split, consensus = test_params
    """
    __default_options = {
        'multiprocess': False,      # whether to use multiprocessing
        'nproc':        4,          # number of processes to use if multiprocess
        'problem':      'primal',   # problem form to solve (UNUSED)
        'consensus':    'simple',   # consensus method to use (simple or general)
        'split':        'naive',    # partition method
        'solver':       'ecos',     # solver to use
        'N':            1,          # number of subsystems
        'max iters':    100,        # number of admm iterations
        'rho':          1,          # rho parameter for admm
        'show spy':     False       # UNUSED
    }
    """
    options = {'N': N, 'max iters': runs, 'rho': 2, 'multiprocess': False, 
               'split': split, 'consensus': consensus}
    results[label] = consensus_conic_opt.solve(gp.socp_vars, options)

def objective(x):
    return gp.socp_vars['c'].dot(x)

print "central objval", objective(gp.socp_sol)
for k in tests:
    print k, objective(results[k]['sol'])

import pylab
lines = []
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
styles = ['-', ':']

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



