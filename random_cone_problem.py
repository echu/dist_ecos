import numpy as np
import scipy.sparse as sp

# size of each block
block_m,block_n = 100, 20

# number of blocks
N = 500

# sparsity density of coupling blocks
p = 0.0005/N

# add blocks to the diagonal
diags = []
for i in xrange(N):
    delta_m, delta_n = np.random.randint(-10,10), np.random.randint(-2, 2)
    diags.append( np.random.randn(block_m + delta_m, block_n + delta_n) )

# create the sparse diagonal matrix
A = sp.block_diag(diags).tocsr()

# create sparse coupling
m, n = A.shape
nnz = m*n*p
ij = (np.random.randint(m, size=(nnz,)), np.random.randint(n, size=(nnz,)))
data = np.random.randn(nnz)
Acouple = sp.coo_matrix((data, ij), (m, n))

# add the two
A = A + Acouple

# randomly generate an "x" (free)
x = np.random.randn((n))

# generate a vector which will be used to create the complementary "s" and "y"
tmp = np.random.randn((m))
y = np.maximum(tmp,0)
s = np.maximum(-tmp,0)

# now, s and y are complementary primal-dual pair

c = -A.T*y
b = A*x + s


dims = {'l': m, 'q': [], 's': []}
# A = A.tocoo()
# solvers.conelp(cvxopt.matrix(c), cvxopt.spmatrix(A.data, A.row, A.col), cvxopt.matrix(b), dims)
# 
import ecos
ecos.solve(c, A, b, dims)

objval = c.T.dot(x)
socp_vars = {'c':c, 'G':A, 'h':b, 'A': None, 'b': None, 'dims':dims}
#print b.T.dot(y)
