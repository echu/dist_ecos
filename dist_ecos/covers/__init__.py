""" Return a list describing which rows are assigned to which
    subsystem. The list should have length equal to the
    number of equality constraints (A) plus the number of
    cones in G. We think of compressing each cone into a
    single row for parititioning purposes.
"""
