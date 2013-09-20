from . consensus.general import make_GC_split
from . consensus.simple import make_SC_split
from . covers import metis, naive, random

GC_split = make_GC_split(naive.cover)
GC_metis_split = make_GC_split(metis.cover)
GC_random_split = make_GC_split(random.cover)

SC_split = make_SC_split(naive.cover)
SC_metis_split = make_SC_split(metis.cover)
SC_random_split = make_SC_split(random.cover)


