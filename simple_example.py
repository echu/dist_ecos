#!/usr/bin/env python
"""Run an example of simple consensus"""
import numpy as np
import dist_ecos.split as split
from dist_ecos.covers.helpers import show_spy
from dist_ecos.admm import ADMM

#'global' problem
import dist_ecos.problems.svc as gp

print gp.socp_vars
show_spy(gp.socp_vars)

ADMM.settings['max_iters'] = 1000
ADMM.settings['show_spy'] = True

#simple consensus
simple = ADMM(5, split.SC_split)
diffs = simple.solve(gp)

import pylab
r = len(diffs)
pylab.semilogy(range(r), diffs)
pylab.show()
