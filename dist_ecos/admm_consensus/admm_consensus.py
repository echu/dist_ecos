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


#we define these methods of the AdmmAgent outside of the class so that
#they can be pickled to be mapped over agents. kind of annoying, but it works
# for now
def prox(self, z):
    """ compute admm update, which involves the prox function
    """

    #TODO: should update this later with correct rho info
    t = time.time()
    z = z[self.local_vars]
    self.x = self.prox_obj.prox(z - self.u)
    prox_time = time.time() - t

    result = {'x': self.x,
              'index': self.local_vars,
              'time': prox_time}

    return result


def update_dual(self, z, zold):
    z = z[self.local_vars]
    zold = zold[self.local_vars]

    offset = self.x - z
    self.u += offset

    # use primal and dual residuals for *cone* program
    #subprob = self.prox_obj.subproblem
    # computed slack vars
    #s = self.prox_obj.s
    
    # z is in the cone, check b - A*z and h - G*z - s
    #primal_term = np.linalg.norm(subprob['h'] - subprob['G'] * z - s)**2
    #if subprob['A'] is not None:
    #    primal_term += np.linalg.norm(subprob['b'] - subprob['A'] * z)**2
    
    primal_term = np.linalg.norm(offset)**2
    
    # there's no stopping condition that corresponds to dual; all i can
    # say is that we're attempting to find a fixed point so that there exists
    # a y which is optimal when `dual_term` is small...
    #
    # XXX: need to make above statement more precise.
    dual_term = np.linalg.norm(self.rho*(z - zold))**2
        
    return {'primal': primal_term, 'dual': dual_term}


def reset_dual(self):
    self.u = np.zeros((self.n))


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


class AgentList(object):
    def __init__(self, prox_list, local_var_list, rho, parallel, n):
        self.n = n
        #wraps Admm state information around an object which computes a prox
        self.agents = [AdmmAgent(prox, local_vars, rho) for prox, local_vars in izip(prox_list, local_var_list)]
        #wraps admmAgent with a process to perform the work in parallel
        self.agents = StatefulMapper(self.agents, parallel=parallel)

        #compute the var count for computing average
        self.var_count = np.zeros(self.n)
        for local_vars in local_var_list:
            self.var_count[local_vars] += 1

    def update(self, z):
        # admm step of prox(z-u) with whatever eacy subsystem's current u is
        results = self.agents.map(prox, z)
        times = []

        zold = z
        z = np.zeros(self.n)
        for result in results:
            z[result['index']] += result['x']
            times.append(result['time'])

        z = z/self.var_count

        results = self.agents.map(update_dual, z, zold)

        pri, dual = 0.0, 0.0
        for result in results:
            pri += result['primal']
            dual += result['dual']

        return z, math.sqrt(pri), math.sqrt(dual), times

    def reset(self):
        self.agents.map(reset_dual)

    def close(self):
        self.agents.close()


def solve(prox_list, local_var_list, parallel=True, max_iters=100, rho=1, restart=False, backtrack=False, x0=[]):
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
    N = len(prox_list)

    #leave the rho computation to the prox objects for now. fix this later

    #still need rho for the correct residual computation...
    if x0 == []:
        z = np.zeros((n))
    else:
        z = x0

    res_pri = []
    res_dual = []
    zs = []

    prox_times = [[] for i in range(N)]

    #initialize each admm agent
    agent_list = AgentList(prox_list, local_var_list, rho, parallel, n)

    #admm iteration
    t = time.time()
    for i in xrange(max_iters):
        print '>> iter %d' % i

        zold = z
        # updates their local x and dual var
        z, pri, dual, prox_step_time = agent_list.update(z)

        for j, subsystem_time in enumerate(prox_step_time):
            prox_times[j].append(subsystem_time)

        if i == 0 or pri <= res_pri[-1]:
            #then good step, update everything
            #print 'good step'
            pass
        else:
            #print 'bad step'
            #bad step, reset
            if restart:
                #reset the dual variables to zero
                agent_list.reset()

                if backtrack:
                    #reset z to the center of the xs with the smallest variance
                    z = zold

        zs.append(z)
        res_pri.append(pri)
        res_dual.append(dual)
        
        if pri < 1e-1 and dual < 1e-1:
            break

    solve_time = time.time() - t

    subsystem_stats = [(min(x), max(x)) for x in prox_times]

    #close the agent list
    agent_list.close()

    result = {'res_pri': res_pri,
              'res_dual': res_dual,
              'sol': z,
              'z_list': zs,
              'solve_time': solve_time,
              'subsystem_stats': subsystem_stats,
              'iters': i}

    return result
