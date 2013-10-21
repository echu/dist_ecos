import collections
import numpy as np
import math
from multiprocessing import Pool


def xupdate(args):
    """ performs the mapping step
    """
    p, z, state = args
    state, info = p.xupdate(z[p.global_index], state)
    return {
        'state': state,
        'index': p.global_index,
        'primal': info['primal'],
        'dual': info['dual']
    }


class ProxList(collections.Iterable):

    def __init__(self, n, proxes, multiprocess=False, nproc=4, **kwargs):
        self.n = n              # number of variables
        self.proxes = proxes    # prox operators

        if multiprocess:
            self.nproc = nproc
            self.map = Pool(processes=nproc).map
        else:
            self.nproc = 1
            self.map = map

        self.z_count = np.zeros((n))

        # initialize the number of subsystems that each variable is in
        for p in proxes:
            self.z_count[p.global_index] += 1

        self.results = None

    def __iter__(self):
        return iter(self.proxes)

    def update(self, z):
        self.results = self.map(
            xupdate, ((p, z, p.state) for p in self.proxes))
        # update the prox state
        for p, result in zip(self.proxes, self.results):
            p.state = result['state']

    def reduce(self):
        if self.results:
            z_total = np.zeros((self.n))
            pri_resid, dual_resid = 0, 0

            # accumulate the new z values
            for result in self.results:
                z_total[result['index']] += result['state'].x
                pri_resid += result['primal']
                dual_resid += result['dual']

            # record new residuals and take an average
            resids = {
                'primal': math.sqrt(pri_resid),
                'dual':   math.sqrt(dual_resid)
            }
            z = z_total / self.z_count
            return z, resids
        raise Exception("No results to reduce!")

# class TestProx(object):
#     def __init__(self, x):
#         self.x = x
#         self.global_index = np.arange(3)
#
#     def xupdate(self, z):
#         info = {'primal': 1, 'dual': 0}
#         return self.x + z, info
#
# a = TestProx(np.array([1.,2.,3.]))
# b = TestProx(np.array([2.,3.,1.]))
# c = TestProx(np.array([3.,1.,2.]))
#
# p = ProxList(3, [a,b,c], True)
#
# p.update(np.array([1,1.,1.]))
# print p.z_count
# print p.reduce()
