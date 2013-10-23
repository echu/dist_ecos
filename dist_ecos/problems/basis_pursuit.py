import numpy as np
from qcml import QCML
import ecos

m = 10
n = 20
A = np.random.randn(m, n)
b = np.random.randn(m)

q = QCML()

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

socp_vars = q.prob2socp(locals())

# convert to CSR for fast row slicing to distribute problem
socp_vars['A'] = socp_vars['A'].tocsr()
socp_vars['G'] = socp_vars['G'].tocsr()

# the size of the stuffed x or v
n = socp_vars['A'].shape[1]

ecos_sol = ecos.solve(**socp_vars)

# solution to transformed socp (stuffed)
socp_sol = ecos_sol['x']
# solution to original problem (unstuffed)
prob_sol = q.socp2prob(ecos_sol['x'])['x']
