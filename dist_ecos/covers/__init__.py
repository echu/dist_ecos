""" A module of covers / partitions.

    Each function must implement the prototype:

        list_of_equations = cover_func(R, s, cone_array, num_partitions)

    These functions take in R-format data and output a list of R-format data
    representing a cover or split of the equations.
    Note that the objective term, if one exists, still needs to be handled by
    some outside function.

    TODO: These functions could probably be further refactored, since only
    the selection indices are important.
"""
