"""Compare simple consensus vs general consensus on the same problem"""
import numpy as np
import split


#'global' problem
import problems.svc as gp

print gp.socp_vars
split.show_spy(gp.socp_vars)

n = gp.n
num_proxes = 8
runs = 1000

#simple consensus
z = np.zeros((n))
diffs_simple = []

proxes = split.SC_split(gp.socp_vars, num_proxes)

# for p in proxes:
#     split.show_spy(p.socp_vars)

for j in xrange(runs):
    print 'iter %d' % j

    total = np.zeros((n))

    for p in proxes:
        x, info = p.xupdate(z)
        total += x

    z_old = z
    z = total/num_proxes

    diffs_simple.append(np.linalg.norm(np.array(z-gp.socp_sol)))
    
#simple consensus with metis
z = np.zeros((n))
diffs_simple_metis = []

proxes = split.SC_metis_split(gp.socp_vars, num_proxes)

# for p in proxes:
#     split.show_spy(p.socp_vars)

for j in xrange(runs):
    print 'iter %d' % j

    total = np.zeros((n))

    for p in proxes:
        x, info = p.xupdate(z)
        total += x

    z_old = z
    z = total/num_proxes

    diffs_simple_metis.append(np.linalg.norm(np.array(z-gp.socp_sol)))


#general consensus
z = np.zeros((n))
diffs_general = []

proxes, c_count = split.GC_split(gp.socp_vars, num_proxes)

# for p in proxes:
#     split.show_spy(p.socp_vars)

for j in xrange(runs):
    print 'iter %d' % j

    total = np.zeros((n))

    for p in proxes:
        x, info = p.xupdate(z[p.global_index])
        total[p.global_index] += x

    z_old = z
    z = total/c_count

    diffs_general.append(np.linalg.norm(np.array(z-gp.socp_sol)))

#general consensus with metis
z = np.zeros((n))
diffs_general_metis = []

proxes, c_count = split.GC_metis_split(gp.socp_vars, num_proxes)

for p in proxes:
    split.show_spy(p.socp_vars)

for j in xrange(runs):
    print 'iter %d' % j

    total = np.zeros((n))

    for p in proxes:
        x, info = p.xupdate(z[p.global_index])
        total[p.global_index] += x

    z_old = z
    z = total/c_count

    diffs_general_metis.append(np.linalg.norm(np.array(z-gp.socp_sol)))


import pylab
pylab.semilogy(
    range(runs), diffs_simple, 
    range(runs), diffs_simple_metis, 
    range(runs), diffs_general, 
    range(runs), diffs_general_metis
)
pylab.legend(['simple', 'simple with metis', 'general', 'general with metis'])
pylab.show()
