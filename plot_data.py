import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

files = [
"2014-01-02-25x5x256.p",
"2014-01-02-50x10x256.p",
"2014-01-02-250x50x256.p",
"2014-01-02-500x100x256.p",
"2014-01-02-2500x500x256.p"
]

block_sizes, ecos_times, admm_times = [], [], []
for filename in files:
    data = pickle.load(open(filename))
    block_sizes.append(data['block_m'])
    ecos_times.append(data['ecos'])
    admm_times.append(data['admm'])
    print data['rel_err']

plt.loglog(block_sizes, ecos_times, block_sizes, admm_times)
plt.xlabel('average block row size')
plt.ylabel('solve time')
plt.legend(['ecos', 'admm'])

plt.show()

print ecos_times
print admm_times