#test_mapper.py
from stateful_mapper2 import StatefulMapper


class Display(object):
    def show(self, x):
        y = x+x
        y = y+1


def do_something(x):
    x.show(10)

n = 2
iters = 10000

objs = [Display() for i in range(n)]

mapper = StatefulMapper(objs, parallel=True)

for i in range(iters):
    print '>> iter: ', i
    mapper.map(do_something)

mapper.close()
