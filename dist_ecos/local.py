from qcml import QCML
import numpy as np
import ecos
import scipy.sparse as sp


def primal2dual(socp_vars):
    '''takes a set of socp_vars in ecos format and outputs
        a set of socp_vars for the dual, also in ecos format
    '''
    w_len = len(socp_vars['b'])
    z_len = len(socp_vars['h'])
    A = sp.hstack([socp_vars['A'].T, socp_vars['G'].T], 'csr')
    c = np.hstack([socp_vars['b'], socp_vars['h']])
    b = -socp_vars['c']
    G = sp.hstack([sp.coo_matrix((z_len, w_len)), -1*sp.identity(z_len)],
                  'csr')
    h = np.zeros((z_len))

    dual_socp_vars = {'A': A, 'G': G, 'c': c, 'b': b, 'h': h,
                      'dims': socp_vars['dims']}

    return dual_socp_vars


def start_row(n, k, i):
    '''start of ith partition of list of length n into k partitions'''
    d = n//k
    r = n % k
    return d*i + min(i, r)


def split_both(socp_vars, total_num, i, rho=1):
    '''return local problem i of total_num.
    A and G matrices should probably be in a sparse format
    which does row slices quickly, like CSR'''
    local_socp_vars = {}
    local_socp_vars['dims'] = socp_vars['dims']
    local_socp_vars['c'] = socp_vars['c']

    m, n = socp_vars['A'].shape
    start = start_row(m, total_num, i)
    stop = start_row(m, total_num, i+1)
    local_socp_vars['A'] = socp_vars['A'][start:stop, :]
    local_socp_vars['b'] = socp_vars['b'][start:stop]

    m, n = socp_vars['G'].shape
    start = start_row(m, total_num, i)
    stop = start_row(m, total_num, i+1)
    local_socp_vars['G'] = socp_vars['G'][start:stop, :]
    local_socp_vars['h'] = socp_vars['h'][start:stop]

    return LocalProblem(local_socp_vars, rho)


def split_A(socp_vars, total_num, i, rho=1):
    '''return local problem i of total_num.
    A matrix is split.
    full G matrix is given to each
    A should probably be in a sparse format
    which does row slices quickly, like CSR'''
    local_socp_vars = {}
    local_socp_vars['dims'] = socp_vars['dims']
    local_socp_vars['c'] = socp_vars['c']

    m, n = socp_vars['A'].shape
    start = start_row(m, total_num, i)
    stop = start_row(m, total_num, i+1)
    local_socp_vars['A'] = socp_vars['A'][start:stop, :]
    local_socp_vars['b'] = socp_vars['b'][start:stop]

    local_socp_vars['G'] = socp_vars['G']
    local_socp_vars['h'] = socp_vars['h']

    return LocalProblem(local_socp_vars, rho)


def show_spy(socp_vars):
    import pylab

    pylab.figure(1)
    pylab.subplot(211)
    pylab.spy(socp_vars['A'], marker='.')
    pylab.xlabel('A')

    pylab.subplot(212)
    pylab.spy(socp_vars['G'], marker='.')
    pylab.xlabel('G')

    pylab.show()


def shuffle_rows(A, b):
    m, n = A.shape
    p = np.random.permutation(m)
    A = A.copy()
    b = b.copy()

    A = A[p, :]
    b = b[p]

    return A, b


class LocalProblem(object):
    '''object to hold local data to compute prox operators of sub problems'''
    def __init__(self, local_socp_vars, rho=1):
        '''local_socp_vars is a subset of the global socp, typically a
        subset of the rows in the equality constraints, but the whole 'c'
        vector, and the whole 'dims' dict

        rho is the admm parameter

        I assume G*x <= h is just linear cones. no quadratic!'''
        self.socp_vars = local_socp_vars
        self.socp_vars['rho'] = rho

        #now we form the data for computing the prox
        #later, we should be able to call a 'prox' form of ecos
        #and not need to use QCML

        if self.socp_vars['A'] is None:
            ma, nA = 0, 0
        else:
            mA, nA = self.socp_vars['A'].shape

        if self.socp_vars['G'] is None:
            mG, nG = 0, 0
        else:
            mG, nG = self.socp_vars['G'].shape

        #n gives the size of the (stuffed) problem, equivalently,
        #the size of 'v'
        self.n = max(nA, nG)

        q = QCML()
        s = '''
        dimensions mA mG n
        variable x(n)
        parameters A(mA,n) G(mG,n) c(n) b(mA) h(mG) v(n)
        parameter rho positive
        minimize c'*x + rho/2*square(norm(x-v))
        A*x == b
        G*x <= h
        '''
        q.parse(s)
        q.canonicalize()
        q.dims = {'mA': mA, 'mG': mG, 'n': self.n}
        q.codegen('python')

        self.q = q

        #local primal and dual admm variables
        #we could have user input here, but we'll default to zero for now
        self.x = np.zeros((self.n))
        self.u = np.zeros((self.n))

    def prox(self, v):
        '''computes prox_{f/rho}(v)'''
        self.socp_vars['v'] = v

        #now we have v, so we can stuff the matrices
        prox_socp_vars = self.q.prob2socp(self.socp_vars)

        #call ecos to produce the prox
        sol = ecos.solve(**prox_socp_vars)
        return self.q.socp2prob(sol['x'])['x']

    def xupdate(self, xbar):
        offset = self.x - xbar
        self.u += offset
        self.x = self.prox(xbar - self.u)
        info = {'offset': np.linalg.norm(offset)**2}
        return self.x, info


def primal_example():
    gp = GlobalProblem()
    lp = split_A(gp.socp_vars, 2, 0, rho=1)
    #n is size of 'v'
    n = lp.n
    v = np.random.rand(n)
    lp.prox(v)

    show_spy(gp.socp_vars)
    show_spy(lp.socp_vars)


def dual_example():
    gp = GlobalProblem()
    dual_socp = primal2dual(gp.socp_vars)
    lp = split_A(dual_socp, 2, 1, rho=1)
    #n is size of 'v'
    n = lp.n
    v = np.random.rand(n)
    lp.prox(v)

    show_spy(dual_socp)
    show_spy(lp.socp_vars)


def shuffle_example():
    gp = GlobalProblem()
    dual_socp = primal2dual(gp.socp_vars)
    show_spy(dual_socp)

    dual_socp['A'], dual_socp['b'] = shuffle_rows(dual_socp['A'],
                                                  dual_socp['b'])
    show_spy(dual_socp)

    lp1 = split_A(dual_socp, 2, 0, rho=1)
    show_spy(lp1.socp_vars)

    lp2 = split_A(dual_socp, 2, 1, rho=1)
    show_spy(lp2.socp_vars)

if __name__ == '__main__':
    #primal_example()
    #dual_example()
    shuffle_example()
