import pymetis as pm
import networkx as nx
import numpy as np
import scipy.io as io
import scipy.sparse as sp

from . utils import form_laplacian
from .. import settings

def cover(socp_data, N):
    """stacks the socp data and partitions it into N
    local dicts describing constraints R <= s"""
    if not settings.paths['mondriaan']:
        raise Exception("Please provide a path to mondriaan: settings.paths['mondriaan'] = PATH.")
    
    n = socp_data['c'].shape[0]

    # form the Laplacian and use pymetis to partition
    L = form_laplacian(socp_data)
    io.mmwrite("mondriaan.mtx", L)
    
    
    import subprocess
    outpath = "mondriaan.mtx-P%d" % N
    proc = subprocess.Popen([settings.paths['mondriaan'], "mondriaan.mtx", str(N), "0.05"])
    proc.wait()
    
    with open(outpath,"r") as f:
        f.readline()    # ignore comments
        f.readline()    # ignore comments
        
        # basic info about the matrix        
        m,_,_,_ = f.readline().strip().split(" ")    
        pstart = []
        # read the starting index of the partition
        for i in xrange(N+1):
            pstart.append(int(f.readline()))
        part_vert = np.zeros(int(m),dtype=np.int)
        count = 0
        part = 0
        for i in xrange(N):
            while count < pstart[i+1]:
                (row, col, val) = f.readline().strip().split(" ")
                part_vert[int(row)-1] = part
                count += 1
            part += 1
            
    return part_vert[n:]
    