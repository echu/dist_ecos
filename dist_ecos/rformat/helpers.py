""" Helpers for problem formation code
"""


def show_spy(socp_vars):
    '''Show the sparsity pattern of A and G in
    the socp_vars dictionary
    '''
    import pylab

    pylab.figure(1)
    pylab.subplot(211)
    #print 'A is', socp_vars['A']
    if socp_vars['A'] is not None:
        pylab.spy(socp_vars['A'], marker='.')
    pylab.xlabel('A')

    #print 'G is', socp_vars['G']
    pylab.subplot(212)
    pylab.spy(socp_vars['G'], marker='.')
    pylab.xlabel('G')

    pylab.show()
