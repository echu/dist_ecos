import multiprocessing
from qcml import QCML
import numpy as np

class PersistMult(multiprocessing.Process):
    #def __init__(self,c=None,A=None,b=None,G=None,h=None,rho=1):
    def __init__(self,A,in_queue,out_queue):
        '''first, we try matrix multiply'''
        '''sets up prox function'''
        super(PersistMult, self).__init__()
        self.A = A
        self.in_queue = in_queue
        self.out_queue = out_queue
        
    def mult(self,v):
        '''does matrix multiply'''
        '''computes prox_{f/rho}(v)'''
        return self.A.dot(v)
    
    def run(self):
        while True:
            next_task = self.in_queue.get()
            if next_task is None:
                # Poison pill means we should exit
                print '%s: Exiting' % self.name
                break
            answer = self.mult(next_task)
            self.out_queue.put(answer)
            in_queue.task_done()
        return
    
in_queue = multiprocessing.JoinableQueue()
out_queue = multiprocessing.JoinableQueue()

    
proxes = []
mats = []
diffs = []

num_mats = 5
n = 11
for i in range(num_mats):
    A = np.random.randn(n,n)
    A = A + A.T
    mats.append(A)
    
    p = PersistMult(A,in_queue,out_queue)
    p.start()
    proxes.append(p)
    
xbar = np.ones((n))
for j in range(100):
    #print 'iter %d'%j
    for i in range(num_mats):
        in_queue.put(xbar)
    in_queue.join() #syncronize to when the in_queue is empty
    total = np.zeros((n))
    for i in range(num_mats):
        total += out_queue.get()
        out_queue.task_done()
    #out_queue.join()
    xbar_old = xbar
    xbar = total/np.linalg.norm(total)
    xbar = xbar*np.sign(xbar[0])
    diffs.append(np.linalg.norm(xbar-xbar_old))


#stops the proxes from running
for i in range(num_mats):
    in_queue.put(None)

for p in proxes:
    p.join()
    
#for i in range(10):
#    x = out_queue.get()
#    print x

import pylab
pylab.semilogy(diffs)
pylab.show()
