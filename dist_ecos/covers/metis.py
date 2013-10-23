import pymetis as pm
import networkx as nx
import numpy as np
import scipy.sparse as sp
from . utils import form_laplacian


def cover(socp_data, N):
    """stacks the socp data and partitions it into N
    local dicts describing constraints R <= s"""
    n = socp_data['c'].shape[0]

    # form the Laplacian and use pymetis to partition
    L = form_laplacian(socp_data)
    graph = nx.from_scipy_sparse_matrix(L)
    cuts, part_vert = pm.part_graph(N, graph)

    return part_vert[n:]
