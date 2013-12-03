from dist_ecos.admm_consensus.prox_list import form_prox_list
import random_cone_problem_fixed_part as gp
from dist_ecos.admm_consensus.admm_consensus import solve
import pylab
import time

def objective(x):
    return gp.socp_vars['c'].dot(x)

pylab.spy(gp.socp_vars['G'], marker='.', alpha=0.2)
pylab.show()

t = time.time()
prox_list, global_indices = form_prox_list(gp.socp_vars, gp.partition_list)
split_time = time.time() - t

result = solve(prox_list, global_indices, parallel=True, max_iters=50, rho=1.)

pri = result['res_pri']
dual = result['res_dual']

pylab.semilogy(range(len(pri)), pri, range(len(dual)), dual)
pylab.legend(['primal', 'dual'])
pylab.show()

print 'split time: ', split_time
print 'solve time: ', result['solve_time']

print 'subsystem times'
for x in result['subsystem_stats']:
    print 'subsystem: ', x

print
print 'admm objective   ', objective(result['sol'])
print 'optimal objective', gp.objval