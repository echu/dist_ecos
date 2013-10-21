#import problems.basis_pursuit as gp
from .. problems import svc as gp
from .. covers.helpers import show_spy
from .. split import GC_split

print 'gp socpvars is ', gp.socp_vars
show_spy(gp.socp_vars)

proxes, c_count = GC_split(gp.socp_vars, 5)
for p in proxes:
    show_spy(p.socp_vars)
    print p.global_index
print 'c_count is ', c_count
