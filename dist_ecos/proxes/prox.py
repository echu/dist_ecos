import numpy as np
import scipy.sparse as sp
import math
import ecos

from . prox_state import ProxState


def add_quad_regularization(rho, c, G, h, dims, A, b):

    # v is scalar var
    # SOC is 1+1+n, n+2 big
    #
    # original problem
    # min c'*x
    # s.t. A*x == b
    #      G*x + s == h
    #      s in K
    #
    # new prob
    # min c'*x + 1/2*quad_over_lin(diag(rho)*x - diag(rho)*x0,1)
    # s.t. A*x == b
    #      G*x + s == h
    #      s in K
    #
    # conelp form
    # min c'*x + rho/2 * u
    # s.t. A*x == b
    #      G*x + s == h
    #      -u + s1 == 1
    #       u + s2 == 1
    #     -2*diag(rho)*x + s3 == -2*diag(rho)*x0
    #      s in K
    #      (s1, s2, s3) in SOC

    if G is not None:
        m, n = G.shape
    else:
        m, n = 0, c.shape[0]
    c = np.hstack((c, [0.5]))

    # remove the last sum(dim['s']) rows of G
    if G is not None:
        G = G.tocoo()

        Gi = np.hstack((G.row, range(m, m + n + 2)))
        Gj = np.hstack((G.col, [n, n], range(n)))
        Gv = np.hstack((G.data, [-1., 1.], -2. * rho))
    else:
        Gi = np.arange(m, m + n + 2)
        Gj = np.hstack(([n, n], range(n)))
        Gv = np.hstack(([-1., 1.], -2. * rho))

    G = sp.coo_matrix((Gv, (Gi, Gj)), shape=(m + n + 2, n + 1))

    dims['q'].append(n + 2)   # u = quad_over_lin(x - x0)

    if A is not None:
        p = A.shape[0]
        A = sp.hstack((A, sp.coo_matrix(None, shape=(p, 1))))

    if h is not None:
        h = np.hstack((h, np.zeros((n + 2))))
    else:
        h = np.zeros((n + 2))
    h[m:m + 2] = 1.
    G = G.tocsc()
    if A is not None:
        A = A.tocsc()

    return {'c': c, 'G': G, 'h': h, 'A': A, 'b': b, 'dims': dims}


class Prox(object):

    '''object to hold local data to compute prox operators of sub problems'''

    def __init__(self, socp_vars, global_count, global_index=None, solver='ecos', rho=1, **kwargs):
        '''

        rho is the admm parameter'''
        self.n = socp_vars['c'].shape[0]
        if global_index is not None:
            self.global_index = global_index
        else:
            self.global_index = np.arange(self.n)

        private_vars = global_count[self.global_index] == 1
        # why sqrt(rho)?
        rho_vec = math.sqrt(rho) * np.ones((self.n))
        rho_vec[private_vars] = 1e-3   # small regularization on private vars

        self.rho = rho_vec

        # now, add quadratic regularization manually to the problem
        self.socp_vars = add_quad_regularization(rho_vec, **socp_vars)
        self.solver = solver

        self.state = ProxState(self.n)

    def prox(self, v):
        '''computes prox_{f/rho}(v)'''
        # set the RHS of the solver
        if self.n == 0:
            return np.empty(0)

        self.socp_vars['h'][-self.n:] = -2 * self.rho * v
        if self.solver == "ecos":
            sol = ecos.solve(verbose=False, **self.socp_vars)
        else:
            raise Exception('Unknown solver')

        x = np.array(sol['x'])
        x = np.reshape(x, (x.shape[0],))

        return x[:self.n]

    def xupdate(self, z, prox_state):
        # does not use "self.state" to update, since multiprocessing will
        # lose the result anyway; must return the state after updating
        offset = prox_state.x - z
        prox_state.u += offset

        dual = (np.linalg.norm(self.rho * (z - prox_state.zold)) ** 2)
        prox_state.zold = z
        info = {'primal': np.linalg.norm(offset) ** 2, 'dual': dual}

        prox_state.x = self.prox(z - prox_state.u)
        return prox_state, info
