#!/usr/bin/env python
""" One possibility is to fix the *probability* of connection between two
    subsystems. Another is to fix the *number* of connections each subsystem
    has. This is N * prob.

    This files fixes the connection probability. This means that as the
    number of subsystems grows, the number of expected connections also
    grows linearly.
"""
import numpy as np
import scipy.sparse as sp
import argparse

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Generate random block cone programs.')
    parser.add_argument('-m', default = 2500, type=int, dest='m', help="average block height")
    parser.add_argument('-n', default = 500, type=int, dest='n', help="average block width")
    parser.add_argument('-N', default = 256, type=int, dest='N', help="number of blocks")
    parser.add_argument('-p', default = 0.1, type=float, dest='p', help="density of each block")


    args = parser.parse_args()

    #problem parameters
    print "Creating problem with %d blocks, with average block size %d x %d" % (args.N, args.m, args.n)

    # size of each block
    block_m, block_n, block_p = args.m, args.n, args.p #2500, 500, 0.1

    # print "average nnz per row", block_m * block_n * block_p / block_m

    # number of blocks
    N = args.N
    p_coupling = 0.1 * N / block_m #0.002 * N # expected number of nonzeros per row for coupling
    # ave_conn = min(3.,N)
    # p_coupling  = (block_n * N) * (1.0 - np.exp( np.log( 1.- ave_conn / N ) / (block_m * block_n) ) )

    # seed = 10
    # np.random.seed(seed)


    def sprandn(m, n, p):
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


    def get_complementary_pair(x, l, q):
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
        # on average, some percentage of the cones will be lp
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


    # add blocks to the diagonal and generate the cones for each block
    # lindiags contain linear part, qdiags contain soc part
    lindiags, qdiags = [], []
    lin_y, quad_y = [], []
    lin_s, quad_s = [], []
    l, q = 0, []
    lin_part, quad_part = [], []    # keep track of partition / subsystem
    for i in xrange(N):
        m_ub = max(0.1*block_m, 1)
        n_ub = max(0.1*block_n, 1)
        delta_m = np.random.randint(-m_ub, m_ub)
        delta_n = np.random.randint(-n_ub, n_ub)
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
    p = p_coupling / n
    Acouple = sprandn(m, n, p)

    # probability that two subsystems are connected
    number_of_slots = block_m * block_n     # number of chances we have
    nnz_per_row = p_coupling
    nnz = nnz_per_row * block_m
    p_nnz = nnz / (block_m * block_n * N)

    # probability that we are connected is 1 - prob we are not
    p_connected = 1.0 - (1.0 - p_nnz)**(number_of_slots)

    # add the two
    A = A + Acouple

    # randomly generate an "x" (free)
    x = np.random.randn((n))

    # now vstack all the ys and ss to get the full (complementary) y and s
    y = np.reshape(np.hstack(ys), (m,))
    s = np.reshape(np.hstack(ss), (m,))

    # now, s and y are complementary primal-dual pair
    c = -A.T * y
    b = A * x + s


    dims = {'l': l, 'q': q, 's': []}
    # A = A.tocoo()
    # solvers.conelp(cvxopt.matrix(c), cvxopt.spmatrix(A.data, A.row, A.col), cvxopt.matrix(b), dims)
    #
    objval = c.T.dot(x)
    socp_vars = {'c': c, 'G': A, 'h': b, 'A': None, 'b': None, 'dims': dims}

    # print b.T.dot(y)

    def statistics():
        yield 72*"="
        yield "coupling density: %f" % (p,)
        yield "coupling prob:    %f" % (p_connected,)
        yield "average links:    %f" % (p_connected * N,) # includes self
        yield "optimal value:    %f" % objval
        yield "generated gap:    %f" % y.dot(s)
        yield "problem size:     (%d, %d)" % (m, n)
        yield "num blocks:       %d" % N
        yield 72*"="

    for line in statistics():
        print line

    partition = lin_part + quad_part
    """
        at this point, partition is a list with each entry corresponding to a row of G.
        each entry describes which subsystem that row is to be assigned to.
        since cones must stay together, we want to transform this description into a list
        of partitions of rows/cones. each partition in the list will give the number of the *cones*
        which belong to that partition. This allows a single row/cone to belong to more than
        one subsystem.
    """


    #compress partition from rows into 'cones'
    def compress_list(partition, p, l, q):
        """ p is the # of rows in A.
        l, q describe the cones associated with G
        """
        cones = []
        for i in range(p + l):
            cones.append(partition[i])
        row = p+l
        for i in q:
            cones.append(partition[row])
            row += i
        return cones

    partition = compress_list(partition, 0, l, q)

    #get a list of cones for each subsystem
    from collections import defaultdict
    partition_list = defaultdict(list)
    for row, subsystem in enumerate(partition):
        partition_list[subsystem].append(row)

    partition_list = list(partition_list.itervalues())

    # save everything we need into a pickled file

    filename = "data-%dx%dx%d.p" % (block_m, block_n, N)
    print "Saving to file: %s" % filename
    import cPickle as pickle
    data = {'socp_vars': socp_vars,
            'partition_list': partition_list,
            'N': N, 'objval': objval,
            'stats': list(statistics()),
            'block_m': block_m, 'block_n': block_n,
            'prob': p_connected}
    pickle.dump(data, open(filename, "wb"))
