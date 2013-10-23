import numpy as np
from . split import split_problem
import time

__default_options = {
    'multiprocess': False,      # whether to use multiprocessing
    'nproc':        4,          # number of processes to use if multiprocess
    'problem':      'primal',   # problem form to solve (UNUSED)
    'consensus':    'simple',   # consensus method to use (simple or general)
    'split':        'naive',    # partition method
    'solver':       'ecos',     # solver to use
    'N':            1,          # number of subsystems
    'max iters':    100,        # number of admm iterations
    'rho':          1,          # rho parameter for admm
    'show spy':     False       # UNUSED
}

def solve(socp_data, user_options = None):
    """ Solves the cone program stored in `socp_data` using ADMM to
        distribute the cone program. Uses the options in `user_options`.
    """
    n = socp_data['c'].shape[0]
    # set the options for the solver
    options = __default_options
    if user_options:
        options.update(user_options)
    
    # now split the problem
    t = time.time()
    proxes = split_problem(socp_data, options)
    split_time = time.time() - t
    
    # if settings['show_spy']:
    #     for p in proxes:
    #         show_spy(p.socp_vars)
            
    # initialize ADMM
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

    result = {'res_pri': res_pri, 'res_dual': res_dual, 'sol': z[:socp_data['c'].shape[0]], 
                'split_time': split_time, 'solve_time': solve_time}

    return result
    