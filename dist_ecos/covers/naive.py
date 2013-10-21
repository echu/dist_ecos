

def start_row(n, k, i):
    '''start of ith partition of list of length n into k partitions'''
    d = n//k
    r = n % k
    return d*i + min(i, r)


def cover(R, s, cone_array, N):
    """Takes in global Rx <= s format and partitions it into N
    local dicts describing constraints R <= s"""

    m, n = R.shape

    return cover_order(R, s, cone_array, N, range(m))


def cover_order(R, s, cone_array, N, order):
    local_list = []

    m, n = R.shape

    for i in xrange(N):
        local_R_data = {}
        start = start_row(m, N, i)
        stop = start_row(m, N, i+1)

        rows = order[start:stop]

        local_R_data['R'] = R[rows, :]
        local_R_data['s'] = s[rows]
        local_R_data['cone_array'] = [cone_array[i] for i in rows]

        local_list.append(local_R_data)

    return local_list
