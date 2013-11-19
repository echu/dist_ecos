import numpy as np
from . split import split_problem
import time
import intersection_form

__default_options = {
    'multiprocess': False,      # whether to use multiprocessing
    'nproc':        4,          # number of processes to use if multiprocess
    'form':      'primal',   # problem form to solve (primal vs intersect)
    'consensus':    'simple',   # consensus method to use (simple or general)
    'split':        'naive',    # partition method
    'solver':       'ecos',     # solver to use
    'N':            1,          # number of subsystems
    'max iters':    100,        # number of admm iterations
    'rho':          1,          # rho parameter for admm
    'show spy':     False       # UNUSED
}


def solve(socp_data, user_options=None):
    """ Solves the cone program stored in `socp_data` using ADMM to
        distribute the cone program. Uses the options in `user_options`.
    """
    
    # set the options for the solver
    options = __default_options
    if user_options:
        options.update(user_options)

    if user_options['form'] == "intersect":
        socp_data, recover_func = intersection_form.convert(socp_data)
    else:
        recover_func = lambda x: x


    # now split the problem
    t = time.time()
    proxes, ave_coupling = split_problem(socp_data, options)
    split_time = time.time() - t

    # if settings['show_spy']:
    #     for p in proxes:
    #         show_spy(p.socp_vars)

    # initialize ADMM
    n = socp_data['c'].shape[0]
    z = np.zeros((n))

    res_pri = []
    res_dual = []

    # ADMM loop
    t = time.time()
    for i in xrange(options['max iters']):
        print '>> iter %d' % i

        proxes.update(z)  # updates their local x and dual var
        z, resids = proxes.reduce()     # computes bar(z) and also gives resid

        res_pri.append(resids['primal'])
        res_dual.append(resids['dual'])

    solve_time = time.time() - t

    result = {'res_pri': res_pri, 'res_dual': res_dual,
              'sol': recover_func(z[:n]),
              'split_time': split_time, 'solve_time': solve_time,
              'ave_coupling': ave_coupling}

    return result
