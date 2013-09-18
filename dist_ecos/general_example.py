"""Run an example of general form consensus"""
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

proxes, c_count = split.GC_split(gp.socp_vars, 5)

for p in proxes:
    #split.show_spy(p.socp_vars)
    pass

for j in range(runs):
    print 'iter %d' % j

    total = np.zeros((n))

    for p in proxes:
        x, info = p.xupdate(z[p.global_index])
        total[p.global_index] += x

    z_old = z
    z = total/c_count

    diffs.append(np.linalg.norm(np.array(z-gp.socp_sol)))

import pylab
r = len(diffs)
pylab.semilogy(range(r), diffs)
pylab.show()
