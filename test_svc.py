from dist_ecos.admm_consensus.prox_list import form_prox_list
from dist_ecos.problems import svc as gp
import numpy as np
from dist_ecos.admm_consensus.admm_consensus import solve
import pylab

m = 513
k = 4
rows = [list(x) for x in np.array_split(range(m), k)]

prox_list, global_indices = form_prox_list(gp.socp_vars, rows)

result = solve(prox_list, global_indices, parallel=True, max_iters=1000, rho=1, restart=False, backtrack=False)

pri = result['res_pri']
dual = result['res_dual']

pylab.semilogy(range(len(pri)), pri, range(len(dual)), dual)
pylab.legend(['primal', 'dual'])
pylab.show()

print 'distance to sol: ', np.linalg.norm(gp.socp_sol - result['sol'], ord=np.inf)
