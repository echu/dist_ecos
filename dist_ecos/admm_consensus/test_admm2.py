from admm_consensus import solve
import numpy as np
import pylab

n = 5
c = np.random.randn(n)
x0 = np.random.randn(n)

rho = 10.0


class vec_prox(object):
    def prox(self, x0):
        return x0 - c/rho


class ball_proj(object):
    def prox(self, x0):
        vec_len = np.linalg.norm(x0)
        if vec_len <= 1:
            return x0
        else:
            return x0/np.linalg.norm(x0)


class equiv(object):
    """first and second halves of the vector must be equal"""
    def prox(self, x0):
        n = len(x0)
        avg = (x0[:n/2] + x0[n/2:])/2.0
        return np.hstack([avg, avg])

local_var_list = map(np.array,[[0,1,2,3,4],[5,6,7,8,9],[0,1,2,3,4,10,11,12,13,14],[5,6,7,8,9,15,16,17,18,19],[10,11,12,13,14,15,16,17,18,19]])
proxes = [vec_prox(), ball_proj(), equiv(), equiv(), equiv()]

result = solve(proxes, local_var_list,parallel=True,max_iters=1000,rho=rho)

print result['sol']

pri = result['res_pri']
dual = result['res_dual']
pylab.semilogy(range(len(pri)), pri, range(len(dual)), dual)
pylab.legend(['primal','dual'])
pylab.show()

print -1.0*c/np.linalg.norm(c)
