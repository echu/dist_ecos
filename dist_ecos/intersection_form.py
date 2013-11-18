import numpy as np
import scipy.sparse as sp

"""
| c^T | b^T | h^T | =    | 0  |
| A   |   0 | 0   | =    | b  |
| 0   | A^T | G^T | =    | -c |

| G   |   0 | 0   | \leq | h  |
| 0   |   0 | -I  | \leq | 0  |
"""

# we want to return the new socp_data
# a function to recover the x from the solution to this larger system
# assume c has the right size (is not None)


def convert(socp_data):
    """ stuff matrices for the intersection form.
        we take care of the possibility that one of A, G could be None

    """
    c = socp_data['c']
    A = socp_data['A']
    b = socp_data['b']
    G = socp_data['G']
    h = socp_data['h']

    #newA
    rows = []
    rows.append([x for x in [c, b, h] if x is not None])
    if A is not None:
        rows.append([A, None] + ([] if G is None else [None]))
    rows.append([None] + [x.T for x in [A, G] if x is not None])

    newA = sp.bmat(rows)

    #newb
    newb = np.hstack([x for x in [0, b, -c] if x is not None])

    #newG
    if G is not None:
        p, q = G.shape
        if A is None:
            Aspace = []
        else:
            m, n = A.shape
            Aspace = [sp.csr_matrix((p, m))]

        rows = []
        rows.append([G] + Aspace + [None])
        rows.append([None] + Aspace + [-1*sp.eye(p)])
        newG = sp.bmat(rows)

        #newh
        newh = np.hstack([h, np.zeros(p)])

    else:
        newG = None
        newh = None

    #newc should be zero
    newc = np.hstack([np.zeros_like(c)] + [np.zeros(x.shape[0]) for x in [A, G] if x is not None])

    new_socp_data = {'A': newA,
                     'b': newb,
                     'G': newG,
                     'h': newh,
                     'c': newc
                     }

    n = newc.shape[0]  # length of the newc
    p = c.shape[0]  # length of the part of x we want to recover

    def recover(x):
        if np.rank(x) != 1:
            raise Exception("x should be a 1D vector.")
        if x.shape[0] != n:
            raise Exception("x has the wrong length.")

        return x[:n]

    #recovery function

    return new_socp_data, recover
