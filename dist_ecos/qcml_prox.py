import multiprocessing
import numpy as np
from problem import GlobalProblem


class PersistProx(multiprocessing.Process):
    def __init__(self, local_problem, inbox, outbox):
        '''sets up prox function. we assume the cones are linear.
           we expect numpy/scipy data '''
        super(PersistProx, self).__init__()

        self.local_problem = local_problem

        self.inbox = inbox
        self.outbox = outbox

    def run(self):
        while True:
            next_arg = self.inbox.get()
            if next_arg is None:
                # Poison pill means we should exit
                print '%s: Exiting' % self.name
                break
            x, info = self.local_problem.xupdate(next_arg)
            self.outbox.put((x, info))
            inbox.task_done()
        return

if __name__ == '__main__':
    inbox = multiprocessing.JoinableQueue()
    outbox = multiprocessing.JoinableQueue()

    #create a simple l1 problem
    gp = GlobalProblem()
    #n is size of 'v'
    n = gp.n

    proxes = []
    num_prox = 2
    diffs = []

    #primal and dual residuals
    rpris = []
    rduals = []

    xbar = np.zeros((n))

    for i in range(num_prox):
        lp = gp.form_local_prob(num_prox, i, rho=1)
        print lp.local_socp_vars
        p = PersistProx(lp, inbox, outbox)
        #p.start()
        proxes.append(p)

    for j in range(0):
        print 'iter %d' % j
        for i in range(num_prox):
            inbox.put(xbar)
        #syncronize to when the inbox is empty
        inbox.join()
        total = np.zeros((n))

        rpri = 0

        for i in range(num_prox):
            x, info = outbox.get()
            total += x
            rpri += info['offset']
            outbox.task_done()
        outbox.join()

        xbar_old = xbar
        xbar = total/num_prox
        rpri = np.sqrt(rpri)
        rdual = np.sqrt(num_prox)*np.linalg.norm(xbar_old - xbar)

        diffs.append(np.linalg.norm(xbar-gp.socp_sol))
        rpris.append(rpri)
        rduals.append(rdual)

    for j in range(1000):
        print 'iter %d' % j

        rpri = 0

        total = np.zeros((n))

        for p in proxes:
            x, info = p.local_problem.xupdate(xbar)
            #x = p.local_problem.optProx(xbar)
            total += x
            #rpri += info['offset']

        xbar_old = xbar
        xbar = total/num_prox

        #rpri = np.sqrt(rpri)
        rdual = np.sqrt(num_prox)*np.linalg.norm(xbar_old - xbar)

        diffs.append(np.linalg.norm(np.array(xbar-gp.socp_sol)))
        #rpris.append(rpri)
        rduals.append(rdual)

    #stops the proxes from running
    for i in range(num_prox):
        inbox.put(None)

    #for p in proxes:
    #    p.join()

    print diffs

    import pylab
    r = len(diffs)
    #pylab.semilogy(range(r), diffs, range(r), rduals)
    pylab.semilogy(range(r), diffs)
    pylab.show()
