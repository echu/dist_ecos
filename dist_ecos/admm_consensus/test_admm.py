from admm_consensus import solve
import numpy as np

n = 5
c = np.random.randn(n)
x0 = np.random.randn(n)

rho = 10


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

a = vec_prox()
b = ball_proj()

local_var_list = [np.arange(n), np.arange(n)]

result = solve([a, b], local_var_list,parallel=False,max_iters=100,rho=rho)

print result['sol']

print -1.0*c/np.linalg.norm(c)
