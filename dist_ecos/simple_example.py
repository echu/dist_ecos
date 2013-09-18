"""Run an example of simple consensus"""
import numpy as np
import split


#'global' problem
import problems.svc as gp

print gp.socp_vars
split.show_spy(gp.socp_vars)

n = gp.n
z = np.zeros((n))
runs = 1000
diffs = []
num_proxes = 5

proxes = split.SC_split(gp.socp_vars, num_proxes)

for p in proxes:
    split.show_spy(p.socp_vars)

for j in range(runs):
    print 'iter %d' % j

    total = np.zeros((n))

    for p in proxes:
        x, info = p.xupdate(z)
        total += x

    z_old = z
    z = total/num_proxes

    diffs.append(np.linalg.norm(np.array(z-gp.socp_sol)))

import pylab
r = len(diffs)
pylab.semilogy(range(r), diffs)
pylab.show()
