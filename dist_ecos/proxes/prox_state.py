import numpy as np

# storage for the ProxOperator state
class ProxState(object):
    def __init__(self, n):
        #local primal and dual admm variables
        #we could have user input here, but we'll default to zero for now
        self.x = np.zeros((n))
        self.u = np.zeros((n))
        self.zold = np.zeros((n))