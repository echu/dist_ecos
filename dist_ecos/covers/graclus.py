import pymetis as pm
import networkx as nx
import numpy as np
import scipy.sparse as sp

from . utils import form_laplacian

def cover(socp_data, N):
    """stacks the socp data and partitions it into N
    local dicts describing constraints R <= s"""
    n = socp_data['c'].shape[0]

    # form the Laplacian and use graculus to partition
    L = form_laplacian(socp_data)
    graph = nx.from_scipy_sparse_matrix(L)
    
    d = nx.convert.to_dict_of_lists(graph)
    
    edgepath = "graclus.edgelist"
    with open(edgepath, "w") as f:
        f.write("%d %d\n" % (graph.number_of_nodes(), graph.number_of_edges()))
        for k,v in d.iteritems():
            f.write("%d %s\n" % (k+1, ' '.join(map(lambda x: str(x + 1), v))))
    
    import subprocess
    outpath = "graclus.edgelist.part.%d" % N
    proc = subprocess.Popen(["/Users/echu/src/graclus1.2/graclus", edgepath, str(N)])
    proc.wait()
    
    lines = open(outpath,"r").readlines()

    part_vert = []
    for l in lines:
    	 part_vert.append(int(l.strip()))
    
    return part_vert[n:]
   