import numpy as np
import itertools


def cover(socp_data, N):
    """stacks the socp data and partitions it into N
    local dicts describing constraints R <= s"""
    p = socp_data['A'].shape[0] if socp_data['A'] is not None else 0
        
    cone_lengths = socp_data['dims']['l'] + len(socp_data['dims']['q'])
    # alternately assign rows
    infinite_iter = itertools.cycle(range(N))
    return list(itertools.islice(infinite_iter, 0, p + cone_lengths))


