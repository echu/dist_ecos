from qcml import QCML
import numpy as np
import ecos


class GlobalProblem(object):
    def __init__(self):
        m = 10
        n = 20
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        q = QCML()
        self.q = q
        q.parse('''
        dimensions m n
        variable x(n)
        parameters A(m,n) b(m)
        minimize norm1(x)
        A*x == b
        ''')
        q.canonicalize()
        q.dims = {'m': m, 'n': n}
        q.codegen('python')

        self.socp_vars = q.prob2socp(locals())

        #convert to CSR for fast row slicing to distribute problem
        self.socp_vars['A'] = self.socp_vars['A'].tocsr()
        self.socp_vars['G'] = self.socp_vars['G'].tocsr()

        #the size of the stuffed x or v
        self.n = self.socp_vars['A'].shape[1]

        self.ecos_sol = ecos.solve(**self.socp_vars)

        #solution to transformed socp (stuffed)
        self.socp_sol = self.ecos_sol['x']
        #solution to original problem (unstuffed)
        self.prob_sol = q.socp2prob(self.ecos_sol['x'])['x']

    def start_row(self, n, k, i):
        '''start of ith partition of list of length n into k partitions'''
        d = n//k
        r = n % k
        return d*i + min(i, r)

    def form_local_prob(self, total_num, i, rho=1):
        '''return local problem i of total_num'''
        local_socp_vars = {}
        local_socp_vars['dims'] = self.socp_vars['dims']
        local_socp_vars['c'] = self.socp_vars['c']

        m, n = self.socp_vars['A'].shape
        start = self.start_row(m, total_num, i)
        stop = self.start_row(m, total_num, i+1)
        local_socp_vars['A'] = self.socp_vars['A'][start:stop, :]
        local_socp_vars['b'] = self.socp_vars['b'][start:stop]

        m, n = self.socp_vars['G'].shape
        start = self.start_row(m, total_num, i)
        stop = self.start_row(m, total_num, i+1)
        local_socp_vars['G'] = self.socp_vars['G'][start:stop, :]
        local_socp_vars['h'] = self.socp_vars['h'][start:stop]

        return LocalProblem(local_socp_vars, rho)


class LocalProblem(object):
    '''object to hold local data to compute prox operators of sub problems'''
    def __init__(self, local_socp_vars, rho=1):
        '''local_socp_vars is a subset of the global socp, typically a
        subset of the rows in the equality constraints, but the whole 'c'
        vector, and the whole 'dims' dict

        rho is the admm parameter

        I assume G*x <= h is just linear cones. no quadratic!'''
        self.local_socp_vars = local_socp_vars
        self.local_socp_vars['rho'] = rho

        #now we form the data for computing the prox
        #later, we should be able to call a 'prox' form of ecos
        #and not need to use QCML

        if local_socp_vars['A'] is None:
            ma, nA = 0, 0
        else:
            mA, nA = local_socp_vars['A'].shape

        if local_socp_vars['G'] is None:
            mG, nG = 0, 0
        else:
            mG, nG = local_socp_vars['G'].shape

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
        self.local_socp_vars['v'] = v

        #now we have v, so we can stuff the matrices
        prox_socp_vars = self.q.prob2socp(self.local_socp_vars)

        #call ecos to produce the prox
        sol = ecos.solve(**prox_socp_vars)
        return self.q.socp2prob(sol['x'])['x']

    def xupdate(self, xbar):
        offset = self.x - xbar
        self.u += offset
        self.x = self.prox(xbar - self.u)
        info = {'offset': np.linalg.norm(offset)**2}
        return self.x, info

if __name__ == '__main__':
    gp = GlobalProblem()
    lp = gp.form_local_prob(3, 0, 1)
    #n is size of 'v'
    n = lp.n
    v = np.random.rand(n)
    lp.prox(v)
