import numpy as np
import dist_ecos.problems.basis_pursuit as bp
import dist_ecos.local as local

n = bp.n

num_proxes = 2
proxes = []
runs = 1000

xbar = np.zeros((n))

diffs = []
rpris = []
rduals = []

for i in range(num_proxes):
    lp = local.split_both(bp.socp_vars, num_proxes, i, rho=1)
    proxes.append(lp)

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

    diffs.append(np.linalg.norm(np.array(xbar-bp.socp_sol)))
    rpris.append(rpri)
    rduals.append(rdual)

import pylab
r = len(diffs)
pylab.semilogy(range(r), diffs, range(r), rpris, range(r), rduals)
pylab.legend(['dist', 'primal', 'dual'])
pylab.show()
