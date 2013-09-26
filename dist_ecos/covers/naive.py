

def start_row(n, k, i):
    '''start of ith partition of list of length n into k partitions'''
    d = n//k
    r = n % k
    return d*i + min(i, r)


def cover(R, s, cone_array, N):
    """Takes in global Rx <= s format and partitions it into N
    local dicts describing constraints R <= s"""

    local_list = []

    m, n = R.shape

    for i in xrange(N):
        local_R_data = {}
        start = start_row(m, N, i)
        stop = start_row(m, N, i+1)

        local_R_data['R'] = R[start:stop, :]
        local_R_data['s'] = s[start:stop]
        local_R_data['cone_array'] = cone_array[start:stop]

        local_list.append(local_R_data)

    return local_list
