'''Chebyshev centering problem.
the stuffed matrices end up being completley dense'''

import numpy as np
from qcml import QCML
import ecos

m = 40
n = 2

A = np.zeros((m, n))
b = np.zeros(m)

for i in range(m):
    a = np.random.randn(n)
    a = a / np.linalg.norm(a)
    bi = 1 + np.random.rand(1)

    A[i, :] = a
    b[i] = bi

q = QCML()
q.parse('''
        dimensions m n
        variables x(n) r
        parameters A(m,n) b(m)
        maximize r
        A*x + r <= b
        ''')
q.canonicalize()
q.dims = {'m': m, 'n': n}
q.codegen("python")
socp_data = q.prob2socp(locals())

# stuffed variable size
n = socp_data['G'].shape[1]

ecos_sol = ecos.solve(**socp_data)
socp_sol = ecos_sol['x']

prob_sol_x = q.socp2prob(ecos_sol['x'])['x']
prob_sol_r = q.socp2prob(ecos_sol['x'])['r']
