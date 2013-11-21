""" module to compute simple/general consensus with ADMM

    input:  list of objects which compute a prox function. each object
            should implement a .prox(x,rho) function.
            rho can be left out, a scalar, or a vector.
            we allow rho to possibly be a vector to handle
            private variables in general consensus.

            list of mappings from local to global variables.

    output: list of primal and dual residuals for each iteration
            actual iterates
            timing info

    options:    parallel or not
                max_iters
                rho
"""

#let's try to design this in a way that a user can use it if they have a function
#to compute the prox for a fixed rho, scalar rho, and vector rho
#more flexibility gives us more we can do with the algorithms (like hide private variables)
#but we can fall back to simple consensus if need be

import numpy as np
import time
from itertools import izip
from stateful_mapper import StatefulMapper
import math


class AdmmAgent(object):
    """ class to wrap ADMM state information around an object which computes
        a prox
    """
    def __init__(self, prox_obj, local_vars, rho):
        #just leave rho as a scalar for now
        self.prox_obj = prox_obj
        self.rho = rho
        self.local_vars = local_vars
        self.n = len(local_vars)
        self.x = np.zeros((self.n))
        self.u = np.zeros((self.n))
        self.zold = np.zeros((self.n))

    def update(self, z):
        """ compute admm update, which involves the prox function
            update admm state information
            return result of prox and residual information
        """
        z = z[self.local_vars]
        offset = self.x - z
        self.u += offset

        #fix residual computation
        dual = (np.linalg.norm(self.rho * (z - self.zold)) ** 2)
        self.zold = z

        #TODO: should update this later with correct rho info
        self.x = self.prox_obj.prox(z - self.u)

        result = {'primal': np.linalg.norm(offset) ** 2,
                  'dual': dual,
                  'x': self.x,
                  'index': self.local_vars}

        return result


def xupdate(agent, z):
    return agent.update(z)


class AgentList(object):
    def __init__(self, prox_list, local_var_list, rho, parallel, n):
        self.n = n
        #wraps Admm state information around an object which computes a prox
        self.agents = [AdmmAgent(prox, local_vars, rho) for prox, local_vars in izip(prox_list, local_var_list)]
        #wraps admmAgent with a process to perform the work in parallel
        self.agents = StatefulMapper(self.agents, parallel=parallel)
        self.results = None

        #compute the var count for computing average
        self.var_count = np.zeros(self.n)
        for local_vars in local_var_list:
            self.var_count[local_vars] += 1

    def update(self, z):
        self.results = self.agents.map(xupdate, z)

    def reduce(self):
        z_total = np.zeros(self.n)
        pri_resid, dual_resid = 0.0, 0.0
        if self.results is not None:
            for result in self.results:
                pri_resid += result['primal']
                dual_resid += result['dual']
                z_total[result['index']] += result['x']

            z = z_total/self.var_count
            resids = {'primal': math.sqrt(pri_resid),
                      'dual': math.sqrt(dual_resid)}
            return z, resids
        else:
            raise Exception("Need results to reduce on!")

    def close(self):
        self.agents.close()

def solve(prox_list, local_var_list, parallel=True, max_iters=100, rho=1, restart=False):
    #NOTE! right now, rho doesn't do anything but scale the residuals
    #i'm still using old proxes here, which are just prox(x0). we need to switch to
    #prox(x0,rho_vec)

    """
    var_map is a list of numpy vectors. each vector tells us which
    global variable each local variable of a subsystem corresponds to.

    """
    #test to make sure the var_map we are given is correct
    all_vars = reduce(np.union1d, local_var_list, np.empty(0, dtype=int))
    n = len(all_vars)  # size of the global variable
    if not np.all(all_vars == np.arange(n)):
        raise Exception("The local to global map doesn't add up correctly.")

    #leave the rho computation to the prox objects for now. fix this later

    #still need rho for the correct residual computation...
    z = np.zeros((n))
    res_pri = []
    res_dual = []
    zs = []

    #initialize each admm agent
    agent_list = AgentList(prox_list, local_var_list, rho, parallel, n)

    #admm iteration
    t = time.time()
    for i in xrange(max_iters):
        print '>> iter %d' % i

        agent_list.update(z)  # updates their local x and dual var
        z, resids = agent_list.reduce()  # computes bar(z), also gives resid

        zs.append(z)
        res_pri.append(resids['primal'])
        res_dual.append(resids['dual'])

    solve_time = time.time() - t

    #close the agent list
    agent_list.close()

    result = {'res_pri': res_pri,
              'res_dual': res_dual,
              'sol': z,
              'z_list': zs,
              'solve_time': solve_time}

    return result
