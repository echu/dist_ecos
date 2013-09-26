""" Functions for converting to and from ECOS/socp_data format to R-format,
    which is of the form Rx <= s, where the cones related to the generalized
    inequality are stored in some array cone_array. These cones can include
    the zero cone to represent linear equality constraints.
"""

