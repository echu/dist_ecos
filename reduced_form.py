'''code to form the reduced problem'''
import problem
import numpy as np
import pylab
import scipy.sparse as sp


def reduced(socp_vars):
    A = socp_vars['A'].tocoo()
    #G = socp_vars['G'].tocoo()

    pylab.spy(A)
    pylab.show()

    mapping, cols = np.unique(np.hstack([A.col]), return_inverse=True)


    B = sp.coo_matrix((A.data, (A.row, cols)), shape=(A.shape[0], len(mapping)))

    pylab.spy(B)
    pylab.show()

    return A


if __name__ == '__main__':
    gp = problem.GlobalProblem()

    reduced(gp.socp_vars)
