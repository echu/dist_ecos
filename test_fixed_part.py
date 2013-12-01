from dist_ecos.admm_consensus.prox_list import form_prox_list
import random_cone_problem_fixed_part as gp
from dist_ecos.admm_consensus.admm_consensus import solve
import pylab

pylab.spy(gp.socp_vars['G'], marker='.', precision=2.5, alpha=0.2)
pylab.show()


prox_list, global_indices = form_prox_list(gp.socp_vars, gp.partition_list)

result = solve(prox_list, global_indices, parallel=True, max_iters=200, rho=.1, restart=False, backtrack=False)

pri = result['res_pri']
dual = result['res_dual']

pylab.semilogy(range(len(pri)), pri, range(len(dual)), dual)
pylab.legend(['primal', 'dual'])
pylab.show()
