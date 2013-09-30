from . consensus.general import make_GC_split
from . consensus.simple import make_SC_split
from . covers import metis, naive, random
from . rformat import intersection, standard

GC_split = make_GC_split(naive.cover, standard.convert)
GC_metis_split = make_GC_split(metis.cover, standard.convert)
GC_random_split = make_GC_split(random.cover, standard.convert)

SC_split = make_SC_split(naive.cover, standard.convert)
SC_metis_split = make_SC_split(metis.cover, standard.convert)
SC_random_split = make_SC_split(random.cover, standard.convert)


GC_intersect = make_GC_split(naive.cover, intersection.convert)
GC_metis_intersect = make_GC_split(metis.cover, intersection.convert)
GC_random_intersect = make_GC_split(random.cover, intersection.convert)

SC_intersect = make_SC_split(naive.cover, intersection.convert)
SC_metis_intersect = make_SC_split(metis.cover, intersection.convert)
SC_random_intersect = make_SC_split(random.cover, intersection.convert)


