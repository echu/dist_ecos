import itertools


def cover(socp_data, N):
    """ Return a list describing which rows are assigned to which
        subsystem. The list should have length equal to the
        number of equality constraints (A) plus the number of
        cones in G. We think of compressing each cone into a
        single row for parititioning purposes.

        The rows are dealt out in alternating order.
    """

    p = socp_data['A'].shape[0] if socp_data['A'] is not None else 0

    #compress each second order cone into a single row
    cone_lengths = socp_data['dims']['l'] + len(socp_data['dims']['q'])

    # alternately assign (cone compressed) rows
    infinite_iter = itertools.cycle(range(N))
    return list(itertools.islice(infinite_iter, 0, p + cone_lengths))
