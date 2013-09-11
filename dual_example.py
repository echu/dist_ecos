import numpy as np
import problem

gp = problem.GlobalProblem()
dual_socp_vars = problem.primal2dual(gp.socp_vars)
n = dual_socp_vars['A'].shape[1]
dual_socp_vars['A'], dual_socp_vars['b'] = \
    problem.shuffle_rows(dual_socp_vars['A'], dual_socp_vars['b'])

print dual_socp_vars['A'].shape
print n

problem.show_spy(dual_socp_vars)

num_proxes = 2
proxes = []
runs = 1000

xbar = np.zeros((n))

#diffs = []
rpris = []
rduals = []

for i in range(num_proxes):
    lp = problem.split_A(dual_socp_vars, num_proxes, i, rho=1)
    proxes.append(lp)
    problem.show_spy(lp.socp_vars)


for j in range(runs):
    print 'iter %d' % j

    rpri = 0

    total = np.zeros((n))

    for p in proxes:
        x, info = p.xupdate(xbar)
        total += x
        rpri += info['offset']

    xbar_old = xbar
    xbar = total/num_proxes

    rpri = np.sqrt(rpri)
    rdual = np.sqrt(num_proxes)*np.linalg.norm(xbar_old - xbar)

    #diffs.append(np.linalg.norm(np.array(xbar-gp.socp_sol)))
    rpris.append(rpri)
    rduals.append(rdual)

import pylab
r = len(rpris)
pylab.semilogy(range(r), rpris, range(r), rduals)
pylab.legend(['primal', 'dual'])
pylab.show()
