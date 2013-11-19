def deal(socp_data, N, A_ind, G_ind, linear, soc):
    """ take local subsystem index information and deal it out
        local socp data consisting of full numpy arrays
    """

    socp_datas = []
    indices = []
    for i in xrange(N):
        local_socp_data = {'c': socp_data['c']}
        # A_ind[i] indexes into A, G_ind[i] indexes into G

        # if equality constraints exist and the group has elements
        if socp_data['A'] is not None and A_ind[i]:
            local_socp_data['A'] = socp_data['A'][A_ind[i], :]
            local_socp_data['b'] = socp_data['b'][A_ind[i]]
        else:
            local_socp_data['A'] = None
            local_socp_data['b'] = None

        if G_ind[i] is not None:
            local_socp_data['G'] = socp_data['G'][G_ind[i], :]
            local_socp_data['h'] = socp_data['h'][G_ind[i]]
        else:
            local_socp_data['G'] = None
            local_socp_data['h'] = None
        local_socp_data['dims'] = {'l': linear[i], 'q': soc[i], 's': []}
        indices.append(None)     # global index
        socp_datas.append(local_socp_data)
    return socp_datas, indices
