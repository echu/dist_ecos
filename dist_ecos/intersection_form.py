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

# csr form


def convert(socp_data):
    """ stuff matrices for the intersection form.
        we take care of the possibility that one of A, G could be None

    """
    c = socp_data['c']
    A = socp_data['A']
    b = socp_data['b']
    G = socp_data['G']
    h = socp_data['h']
    dims = socp_data['dims']

    #newA
    rows = []
    rows.append([x for x in [c, b, h] if x is not None])
    if A is not None:
        rows.append([A, None] + ([] if G is None else [None]))
    rows.append([None] + [x.T for x in [A, G] if x is not None])

    newA = sp.bmat(rows, format='csr')

    #newb
    newb = np.hstack([x for x in [0, b, -c] if x is not None])

    #newG
    if G is not None:
        p, q = G.shape
        lin = dims['l']
        quad = sum(dims['q'])
        if A is None:
            Alinspace = []
            Aquadspace = []
        else:
            m, n = A.shape
            Alinspace = [sp.csr_matrix((lin, m))]
            Aquadspace = [sp.csr_matrix((quad, m))]

        rows = []
        if lin > 0 :
            Glin = G[:lin, :]
            hlin = h[:lin]
            rows.append([Glin] + Alinspace + [None])
            rows.append([None] + Alinspace + [-1*sp.eye(lin, p, k=0)])
        else:
            hlin = np.empty(0)
        
        if quad > 0:
            Gquad = G[lin:lin+quad,:]
            hquad = h[lin:lin+quad]
            rows.append([Gquad] + Aquadspace + [None])
            rows.append([None] + Aquadspace + [-1*sp.eye(quad, p, k=lin)])
        else:
            hquad = np.empty(0)
        
        #newG
        newG = sp.bmat(rows, format='csc')

        #newh
        newh = np.hstack([hlin, np.zeros(lin), hquad, np.zeros(quad)])

        #newdims
        newdims = {'l': lin + lin, 'q': dims['q']+dims['q']}
        
        
    else:
        newG = None
        newh = None
        newdims = {'l': 0, 'q': []}

    #newc should be zero
    newc = np.hstack([np.zeros_like(c)] + [np.zeros(x.shape[0]) for x in [A, G] if x is not None])

    new_socp_data = {'A': newA,
                     'b': newb,
                     'G': newG,
                     'h': newh,
                     'c': newc,
                     'dims': newdims
                     }

    n = newc.shape[0]  # length of the newc
    p = c.shape[0]  # length of the part of x we want to recover

    def recover(x):
        if np.rank(x) != 1:
            raise Exception("x should be a 1D vector.")
        if x.shape[0] != n:
            raise Exception("x has the wrong length.")

        return x[:p]

    #recovery function

    return new_socp_data, recover
