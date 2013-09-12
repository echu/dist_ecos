import numpy as np
import dist_ecos.local as local
import multiprocessing

#load the problem data
import dist_ecos.problems.basis_pursuit as bp
from dist_ecos.worker import PersistProx

n = bp.n

num_proxes = 2
proxes = []
runs = 1000

xbar = np.zeros((n))

diffs = []
rpris = []
rduals = []

inbox = multiprocessing.JoinableQueue()
outbox = multiprocessing.JoinableQueue()

for i in range(num_proxes):
    lp = local.split_both(bp.socp_vars, num_proxes, i, rho=1)
    p = PersistProx(lp, inbox, outbox)
    p.start()
    proxes.append(p)

for j in range(1000):
    print 'iter %d' % j
    for i in range(num_proxes):
        inbox.put(xbar)
    #syncronize to when the inbox is empty
    inbox.join()
    total = np.zeros((n))

    rpri = 0

    for i in range(num_proxes):
        x, info = outbox.get()
        total += x
        rpri += info['offset']
        outbox.task_done()
    outbox.join()

    xbar_old = xbar
    xbar = total/num_proxes
    rpri = np.sqrt(rpri)
    rdual = np.sqrt(num_proxes)*np.linalg.norm(xbar_old - xbar)

    diffs.append(np.linalg.norm(xbar-bp.socp_sol))
    rpris.append(rpri)
    rduals.append(rdual)

for i in range(num_proxes):
    inbox.put(None)

for p in proxes:
    p.join()

import pylab
r = len(diffs)
pylab.semilogy(range(r), diffs, range(r), rpris, range(r), rduals)
pylab.legend(['dist', 'primal', 'dual'])
pylab.show()
