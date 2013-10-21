import numpy as np
import scipy.sparse as sp
import math
from multiprocessing import Process
import ecos
import pdos_direct
import cvxopt

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
    
    # TODO: there will be a bug if G is None
    if G is not None:
        m, n = G.shape
    else:
        m, n = 0, c.shape[0]
    c = np.hstack((c,[0.5]))
    
    # remove the last sum(dim['s']) rows of G
    if G is not None:
        G = G.tocoo()

        Gi = np.hstack((G.row,range(m,m+n+2)))
        Gj = np.hstack((G.col,[n,n], range(n)))
        Gv = np.hstack((G.data,[-1.,1.], -2.*rho))
    else:
        Gi = np.arange(m,m+n+2)
        Gj = np.hstack(([n,n], range(n)))
        Gv = np.hstack(([-1.,1.], -2.*rho))
    
    G = sp.coo_matrix((Gv, (Gi, Gj)), shape=(m+n+2,n+1))
    
    dims['q'].append(n+2)   # u = quad_over_lin(x - x0)
    
    if A is not None:
        p = A.shape[0]
        A = sp.hstack((A, sp.coo_matrix(None, shape=(p,n+1))))
    
    if h is not None:
        h = np.hstack((h, np.zeros((n+2))))
    else:
        h = np.zeros((n+2))
    h[m:m+2] = 1.
    G = G.tocsc()
    if A is not None:
        A = A.tocsc()
    
    return {'c': c, 'G': G, 'h':h, 'A': A, 'b':b, 'dims': dims}

class Prox(object):
    '''object to hold local data to compute prox operators of sub problems'''
    def __init__(self, socp_vars, global_count, global_index = None, solver='ecos', rho=1, **kwargs):
        '''local_socp_vars is a subset of the global socp, typically a
        subset of the rows in the equality constraints, but the whole 'c'
        vector, and the whole 'dims' dict

        rho is the admm parameter'''

        self.n = socp_vars['c'].shape[0]
        if global_index is not None:
            self.global_index = global_index
        else:
            self.global_index = np.arange(self.n)
            
        private_vars = global_count[self.global_index] == 1
        rho_vec = math.sqrt(rho)*np.ones((self.n))
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
            
        self.socp_vars['h'][-self.n:] = -2*self.rho*v

        #call ecos to produce the prox
        if self.solver == "ecos":
            # import cvxopt.solvers as solvers
            # import cvxopt
            # sol = solvers.conelp(
            #     cvxopt.matrix(self.socp_vars['c']),
            #     cvxopt.spmatrix(self.socp_vars['G'].tocoo().data, self.socp_vars['G'].tocoo().row, self.socp_vars['G'].tocoo().col),
            #     cvxopt.matrix(self.socp_vars['h']),
            #     self.socp_vars['dims'])
            sol = ecos.solve(**self.socp_vars)
        
        #call pdos to produce the prox
        else:
            opts = {'MAX_ITERS': 100, 'NORMALIZE': True, 'ALPHA': 1.8, 'EPS_ABS': 1e-8,'VERBOSE': False}
            c = cvxopt.matrix(self.socp_vars['c'])
            dims = self.socp_vars['dims']

            if self.socp_vars['b'] is not None:
                b = cvxopt.matrix([
                    cvxopt.matrix(self.socp_vars['b']), 
                    cvxopt.matrix(self.socp_vars['h'])
                ])
                dims['f'] = self.socp_vars['b'].size
            else:
                b = cvxopt.matrix(self.socp_vars['h'])
                dims['f'] = 0
        
            if self.socp_vars['A'] is not None:
                Acoo = self.socp_vars['A'].tocoo()
                Gcoo = self.socp_vars['G'].tocoo()
                m = Acoo.shape[0]
                A = cvxopt.spmatrix(
                    cvxopt.matrix([cvxopt.matrix(Acoo.data), cvxopt.matrix(Gcoo.data)]), 
                    cvxopt.matrix([cvxopt.matrix(Acoo.row), cvxopt.matrix(Gcoo.row+m)]), 
                    cvxopt.matrix([cvxopt.matrix(Acoo.col), cvxopt.matrix(Gcoo.col)])
                )
            else:
                Gcoo = self.socp_vars['G'].tocoo()
                A = cvxopt.spmatrix(Gcoo.data, Gcoo.row, Gcoo.col)
        
            if hasattr(self, 'x0') and hasattr(self, 'y0') and hasattr(self, 's0'):
                sol = pdos_direct.solve(c,A,b,dims, opts, self.x0, self.y0, self.s0)
            else:
                sol = pdos_direct.solve(c,A,b,dims, opts)
                    
            self.x0 = sol['x']
            self.y0 = sol['y']
            self.s0 = sol['s']
        
        #print sol['x']
        #print sol['y']

        x = np.array(sol['x'])
        x = np.reshape(x, (x.shape[0],))
                
        return x[:self.n]

    def xupdate(self, z, prox_state):
        # does not use "self.state" to update, since multiprocessing will
        # lose the result anyway; must return the state after updating
        offset = prox_state.x - z
        prox_state.u += offset

        dual = (np.linalg.norm(self.rho*(z - prox_state.zold))**2)
        prox_state.zold = z
        info = {'primal': np.linalg.norm(offset)**2, 'dual': dual} 

        prox_state.x = self.prox(z - prox_state.u)
        return prox_state, info


if __name__ == '__main__':
    '''linear support vector classifier. This one has no quadratic cone.'''

    import numpy as np
    from qcml import QCML
    import ecos

    n = 16      # number of features
    m = 1024   # number of examples
    X = np.random.randn(m, n) - 1
    Y = np.random.randn(m, n) + 1

    s = """
    dimensions m n
    variable a(n)
    variable b
    parameter X(m,n)      # positive samples
    parameter Y(m,n)      # negative samples
    minimize (0.01*(norm1(a) + abs(b)) + sum(pos(1 - X*a + b) + pos(1 + Y*a - b)))
    """

    p = QCML()
    p.parse(s)
    p.canonicalize()
    p.dims = {'n': n, 'm': m}
    p.codegen("python")

    socp_vars = p.prob2socp(locals())

    #convert to CSR for fast row slicing to distribute problem
    #if socp_vars['A'] is not None:
    #    socp_vars['A'] = socp_vars['A'].tocsr()
    #socp_vars['G'] = socp_vars['G'].tocsr()

    #the size of the stuffed x or v
    n = socp_vars['G'].shape[1]
    print socp_vars
    

    ecos_sol = ecos.solve(**socp_vars)

    #solution to transformed socp (stuffed)
    socp_sol = ecos_sol['x']

    #solution to original problem (unstuffed)
    prob_sol_a = p.socp2prob(ecos_sol['x'])['a']
    prob_sol_b = p.socp2prob(ecos_sol['x'])['b']

    p = Prox(socp_vars, rho=1)
    
    for i in xrange(2):
        p.xupdate(ecos_sol['x'])
   
    #print socp_sol
    
