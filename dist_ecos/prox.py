""" Theoretically, we could implement this prox operator by hand when c = 0.

    TODO: implement by hand if necessary
"""
from qcml import QCML
import numpy as np
import ecos


class Prox(object):
    '''object to hold local data to compute prox operators of sub problems'''
    def __init__(self, socp_vars, rho=1):
        '''local_socp_vars is a subset of the global socp, typically a
        subset of the rows in the equality constraints, but the whole 'c'
        vector, and the whole 'dims' dict

        rho is the admm parameter

        I assume G*x <= h is just linear cones. no quadratic!'''
        self.socp_vars = socp_vars
        self.socp_vars['rho'] = rho
        #self.global_indx = socp_vars['global_indx']

        #now we form the data for computing the prox
        #later, we should be able to call a 'prox' form of ecos
        #and not need to use QCML

        prob_dims = 'dimension n'

        if self.socp_vars['c'] is None:
            prob_param = 'parameters v(n)'
            prob_obj = "minimize norm(x-v)"
        else:
            prob_param = 'parameters c(n) v(n)\nparameter rho positive'
            prob_obj = "minimize c'*x + rho/2*square(norm(x-v))"
            

        prob_var = 'variable x(n)'


        prob_const = ''

        if self.socp_vars['A'] is None:
            mA, nA = 0, 0
        else:
            mA, nA = self.socp_vars['A'].shape
            prob_dims += '\ndimension mA'
            prob_param += '\nparameters A(mA,n) b(mA)'
            prob_const += 'A*x == b\n'

        if self.socp_vars['G'] is None:
            mG, nG = 0, 0
        else:
            mG, nG = self.socp_vars['G'].shape
            prob_dims += '\ndimension mG'
            prob_param += '\nparameters G(mG,n) h(mG)'
            prob_const += 'G*x <= h'

        s = '\n'.join([prob_dims, prob_param, prob_var, prob_obj, prob_const])

        #n gives the size of the (stuffed) problem, equivalently,
        #the size of 'v'
        self.n = max(nA, nG)

        q = QCML()
        try:
            q.parse(s)
        except:
            print s
            raise
        q.canonicalize()
        q.dims = {'mA': mA, 'mG': mG, 'n': self.n}
        q.codegen('python')

        self.q = q

        #local primal and dual admm variables
        #we could have user input here, but we'll default to zero for now
        self.x = np.zeros((self.n))
        self.u = np.zeros((self.n))
        self.zold = np.zeros((self.n))

        self.global_index = np.arange(self.n)

    def prox(self, v):
        '''computes prox_{f/rho}(v)'''
        self.socp_vars['v'] = v

        #now we have v, so we can stuff the matrices
        try:
            prox_socp_vars = self.q.prob2socp(self.socp_vars)
        except:
            print self.q.prob2socp.numbered_source
            raise

        #call ecos to produce the prox
        sol = ecos.solve(**prox_socp_vars)
        return self.q.socp2prob(sol['x'])['x']

    def xupdate(self, z):
        offset = self.x - z
        self.u += offset

        rho = self.socp_vars['rho']
        dual = (rho**2)*(np.linalg.norm(z - self.zold)**2)
        self.zold = z

        self.x = self.prox(z - self.u)
        info = {'primal': np.linalg.norm(offset)**2, 'dual': dual}
        return self.x, info
        
    def restart(self):
        self.u = np.zeros((self.n))


class GCProx(Prox):
    def __init__(self, socp_vars, global_index, rho=1):
        super(GCProx, self).__init__(socp_vars, rho)
        self.global_index = global_index


if __name__ == '__main__':
    import problems.svc as gp

    p = Prox(gp.socp_vars, rho=1)

    print p.socp_vars
