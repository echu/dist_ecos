#!/usr/bin/env python
from dist_ecos.admm_consensus.prox_list import form_prox_list
import random_cone_problem_fixed_part as gp
from dist_ecos.admm_consensus.admm_consensus import solve
import pylab
import time
import cPickle as pickle

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate random block cone programs.')
    parser.add_argument('--infile', type=argparse.FileType('rb'), dest='prob', help="problem to read, pickled file saved by random_cone_prob_fixed_part.py", required=True)
    parser.add_argument('--outfile', default = "timings.p", type=argparse.FileType('wb'), dest='savefile', help="file to save timing information")
    parser.add_argument('--noplot', dest='show_plot', action='store_const', const=False, default=True, help="do not show the plots (if omitted, will display)")

    args = parser.parse_args()
    
    data = pickle.load(args.prob)
    for line in data['stats']:
        print line
    
    socp_vars = data['socp_vars']
    objval = data['objval']
    N = data['N']
    partition_list = data['partition_list']
    
    # first, solve with ecos
    import ecos
    sol = ecos.solve(**socp_vars)
    ecos_time = sol['info']['timing']['runtime']
    
    # next, solve with dist_ecos
    def objective(x):
        return socp_vars['c'].dot(x)

    if args.show_plot:
        pylab.spy(socp_vars['G'], marker='.', alpha=0.2)
        pylab.show()

    t = time.time()
    prox_list, global_indices = form_prox_list(socp_vars, partition_list)
    split_time = time.time() - t

    result = solve(prox_list, global_indices, parallel=True, max_iters=1000, rho=N)

    pri = result['res_pri']
    dual = result['res_dual']

    if args.show_plot:
        pylab.semilogy(range(len(pri)), pri, range(len(dual)), dual)
        pylab.legend(['primal', 'dual'])
        pylab.show()

    print 'split time: ', split_time
    print 'solve time: ', result['solve_time']

    print 'subsystem times'
    for x in result['subsystem_stats']:
        print 'subsystem: ', x

    print
    admm_objval = objective(result['sol'])
    print 'admm objective   ', admm_objval
    print 'optimal objective', objval

    print 
    print 'primal resid:', pri[-1]
    print 'dual resid:  ', dual[-1]
    
    timing = {'ecos': ecos_time, 'admm': result['solve_time'], 
              'block_m': data['block_m'], 'block_n': data['block_n'],
              'prob': data['prob'], 'N': N, 'objval': objval,
              'admm_objval': admm_objval, 'iters': result['iters']+1,
              'rel_err': abs(objval - admm_objval)/abs(objval)}

    pickle.dump(timing, args.savefile)
