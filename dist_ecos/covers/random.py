import numpy as np


def cover(socp_data, N):
    """stacks the socp data and partitions it into N
    local dicts describing constraints R <= s"""
    n = socp_data['c'].shape[0]
    p = socp_data['A'].shape[0] if socp_data['A'] is not None else 0

    cone_lengths = socp_data['dims']['l'] + len(socp_data['dims']['q'])
    random_partition = np.random.randint(0, N, p + cone_lengths)

    return random_partition
