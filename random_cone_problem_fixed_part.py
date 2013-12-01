import numpy as np
import scipy.sparse as sp

def sprandn(m,n,p):
    """ Generates random matrix with entries uniformly distributed and values
        drawn from normal distribution.
    """
    nnz = m * n * p
    ij = (np.random.randint(m, size=(nnz,)), np.random.randint(n, size=(nnz,)))
    data = np.random.randn(nnz)
    return sp.coo_matrix((data, ij), (m, n))

def proj_soc(x, q):
    """ Projects x onto a second-order cone of size "q"
    """
    v = x[1:]
    t = x[0]
    if np.linalg.norm(v) <= -t:
        y = np.zeros(x.shape)
    elif np.linalg.norm(v) <= t:
        y = x
    else:
        alpha = (0.5)*(np.linalg.norm(v) + t)
        y = np.hstack((alpha, alpha*v/np.linalg.norm(v)))
    return y

def get_complementary_pair(x,l,q):
    """ Projects the vector x onto K_l x K_{q_1} x ... x K_{q_n}.
        The input "l" is an integer; "q" is a list of integers.
        
        Returns complementary pair.
    """
    y, s = np.zeros(x.shape), np.zeros(x.shape)
    lin = x[:l]
    y[:l] = np.maximum(lin, 0)
    
    start = l
    for qsize in q:
        quad = x[start:start + qsize]
        y[start:start + qsize] = proj_soc(quad, qsize)
        start = start + qsize
    s = y - x
    return (y, s)

def random_cones(m):
    """ Generates a set of random cones whose total dimension is m.
    """
    # on average, 90% of the cones will be lp
    l = np.random.randint(0.8 * m, m)
    remaining_dims = m - l
    
    # divide all remaining dims into SOC of size "3"
    num_q = remaining_dims // 3
    q = num_q*[3]
    # remainder gets assigned into one large SOC
    remainder = remaining_dims % 3
    if remainder > 0:
        if num_q > 0:
            q[-1] = q[-1] + remainder
        else:
            q = [remainder]
    return (l, q)

# size of each block
block_m, block_n, block_p = 200, 60, 0.3

# print "average nnz per row", block_m * block_n * block_p / block_m

# number of blocks
N = 2

# add blocks to the diagonal and generate the cones for each block
# lindiags contain linear part, qdiags contain soc part
lindiags, qdiags = [], []   
lin_y, quad_y = [], []
lin_s, quad_s = [], []
l, q = 0, []
lin_part, quad_part = [], []    # keep track of partition / subsystem
for i in xrange(N):
    delta_m = np.random.randint(-0.1 * block_m, 0.1 * block_m)
    delta_n = np.random.randint(-0.1 * block_n, 0.1 * block_n)
    delta_p = 2.*(0.1*block_p)*np.random.ranf() - 0.1*block_p
    
    local_m = block_m + delta_m
    local_n = block_n + delta_n
    local_p = block_p + delta_p
    
    # generate a vector which will be used to create the complementary "s" and "y"
    tmp = np.random.randn((local_m))

    # generate random cones and get a complementary pair
    local_l, local_q = random_cones(local_m)
    y, s = get_complementary_pair(tmp, local_l, local_q)
    
    # append all the data
    lin_y.append(y[:local_l])
    lin_s.append(s[:local_l])
    quad_y.append(y[local_l:])
    quad_s.append(s[local_l:])
    
    # append the blocks
    lindiags.append(sprandn(local_l, local_n, local_p))
    qdiags.append(sprandn(local_m - local_l, local_n, local_p))
    
    # count the cone sizes
    l += local_l
    q += local_q
    
    # keep track of partition
    lin_part.extend(local_l*[i])
    quad_part.extend((local_m - local_l)*[i])

ys = lin_y + quad_y
ss = lin_s + quad_s

# create the sparse diagonal matrix
Alin = sp.block_diag(lindiags)
Aquad = sp.block_diag(qdiags)
A = sp.vstack((Alin, Aquad))
m, n = A.shape

# create sparse coupling
# sparsity density of coupling blocks
p = 10.0 / n # average of 50 nnz per row 
Acouple = sprandn(m,n,p)

# add the two
A = A + Acouple

# randomly generate an "x" (free)
x = np.random.randn((n))

# now vstack all the ys and ss to get the full (complementary) y and s
y = np.reshape( np.hstack(ys), (m,) )
s = np.reshape( np.hstack(ss), (m,) )

# now, s and y are complementary primal-dual pair
c = -A.T * y
b = A * x + s


dims = {'l': l, 'q': q, 's': []}
# A = A.tocoo()
# solvers.conelp(cvxopt.matrix(c), cvxopt.spmatrix(A.data, A.row, A.col), cvxopt.matrix(b), dims)
#
import ecos
ecos.solve(c, A, b, dims)

objval = c.T.dot(x)
socp_vars = {'c': c, 'G': A, 'h': b, 'A': None, 'b': None, 'dims': dims}
# print b.T.dot(y)

print 72*"="
print "coupling density:", p
print "optimal value:   ", objval
print "generated gap:   ", y.dot(s)
print "problem size:     (%d, %d)" % (m, n)
print "num blocks:      ", N
print 72*"="

partition = lin_part + quad_part


#compress partition from rows into 'cones'
def compress_list(partition, p, l, q):
    cones = []
    for i in range(p + l):
        cones.append(partition[i])
    row = p+l
    for i in q:
        cones.append(partition[row])
        row += q[i]
    return cones

partition = compress_list(partition, 0, l, q)

#get a list of cones for each subsystem
from collections import defaultdict
partition_list = defaultdict(list)
for row, subsystem in enumerate(partition):
    partition_list[subsystem].append(row)

partition_list = list(partition_list.itervalues())
