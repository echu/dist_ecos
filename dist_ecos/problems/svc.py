'''linear support vector classifier. This one has no quadratic cone.'''

import numpy as np
from qcml import QCML
import ecos

n = 256      # number of features
m = 8092   # number of examples
X = np.random.randn(m, n) - 1
Y = np.random.randn(m, n) + 1

s = """
dimensions m n
variable a(n)
variable b
parameter X(m,n)      # positive samples
parameter Y(m,n)      # negative samples
minimize (0.1*norm(a) + sum(pos(1 - X*a + b) + pos(1 + Y*a - b)))
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

ecos_sol = ecos.solve(**socp_vars)

#solution to transformed socp (stuffed)
socp_sol = ecos_sol['x']

#solution to original problem (unstuffed)
prob_sol_a = p.socp2prob(ecos_sol['x'])['a']
prob_sol_b = p.socp2prob(ecos_sol['x'])['b']

if __name__ == '__main__':
    #maybe add some viz code here
    print socp_vars
