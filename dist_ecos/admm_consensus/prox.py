import numpy as np
import scipy.sparse as sp
import math
import ecos

"""
    code to create a prox object.
    requires the global count of the variables so it can put zero
    regularization on private variables

    I think the c vector needs to already be properly normalized.
    I think the socp vars are already reduced to non-zero variables
"""


def add_quad_regularization(rho, c, G, h, dims, A, b):
    #
    # original problem
    # min c'*x
    # s.t. A*x == b
    #      G*x + s == h
    #      s in K
    #
    # new prob
    # min c'*x + (1/2)||diag(rho^{1/2})*x - diag(rho^{1/2})*x0||_2^2
    # s.t. A*x == b
    #      G*x + s == h
    #      s in K
    #
    # equiv prob
    # min ||x - x0 + diag(1/rho)c||_2
    # s.t. A*x == b
    #      G*x + s == h
    #      s in K
    #
    # conelp form
    # min u
    # s.t. A*x == b
    #      G*x + s == h
    #      -u + s1 == 0
    #      -x + s2 == -x0 + diag(1/rho)c
    #      s in K
    #      (s1, s2) in SOC
    #
    # variable is (x, u)

    if G is not None:
        m, n = G.shape
    else:
        m, n = 0, c.shape[0]
    c = np.zeros((n+1))
    c[n] = 1.

    # TODO: remove the last sum(dim['s']) rows of G
    if G is not None:
        G = G.tocoo()

        Gi = np.hstack((G.row, range(m, m + n + 1)))
        Gj = np.hstack((G.col, [n], range(n)))
        Gv = np.hstack((G.data, [-1.], -np.ones((n))))
    else:
        Gi = np.arange(m, m + n + 1)
        Gj = np.hstack(([n], range(n)))
        Gv = np.hstack(([-1.], -np.ones((n))))

    G = sp.coo_matrix((Gv, (Gi, Gj)), shape=(m + n + 1, n + 1))

    dims['q'].append(n + 1)   # u = norm(...)

    if A is not None:
        p = A.shape[0]
        A = sp.hstack((A, sp.coo_matrix(None, shape=(p, 1))))

    if h is not None:
        h = np.hstack((h, np.zeros((n + 1))))
    else:
        h = np.zeros((n + 1))
    
    G = G.tocsc()
    if A is not None:
        A = A.tocsc()

    return {'c': c, 'G': G, 'h': h, 'A': A, 'b': b, 'dims': dims}


class Prox(object):

    '''object to hold local data to compute prox operators of sub problems'''

    def __init__(self, socp_vars, global_count, global_index=None, solver='ecos', rho=1):
        '''

        rho is the admm parameter'''
        self.n = socp_vars['c'].shape[0]
        if global_index is not None:
            self.global_index = global_index
        else:
            self.global_index = np.arange(self.n)

        #ECHU: private vars are regularized
        #private_vars = global_count[self.global_index] == 1
        #rho_vec = rho * np.ones((self.n))
        #rho_vec[private_vars] = 1e-3   # small regularization on private vars

        self.rho = rho
        # keep track of c offset for prox function
        self.c_offset = socp_vars['c'] / self.rho

        # now, add quadratic regularization manually to the problem
        self.socp_vars = add_quad_regularization(rho, **socp_vars)
        self.solver = solver

    def prox(self, v):
        '''computes prox_{f/rho}(v)'''
        # set the RHS of the solver
        if self.n == 0:
            return np.empty(0)

        self.socp_vars['h'][-self.n:] = -v + self.c_offset
        
        if self.solver == "ecos":
            sol = ecos.solve(verbose=False, **self.socp_vars)
        else:
            raise Exception('Unknown solver')

        x = np.array(sol['x'])
        x = np.reshape(x, (x.shape[0],))

        return x[:self.n]
