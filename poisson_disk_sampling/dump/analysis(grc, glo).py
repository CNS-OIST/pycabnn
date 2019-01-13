import numpy as np
from joblib import Parallel, delayed

def count_neighborhood_grc(x, tree, r):
    temp = tree.query_ball_point(x, r)
    count = np.array([len(x)-1 for x in temp])
    analysis[2*(r-4), 0] = np.mean(count)
    analysis[2*(r-4), 1] = np.var(count)

def count_neighborhood_glo(x, tree, r):
    temp = tree.query_ball_point(x, r)
    count = np.array([len(x)-1 for x in temp])
    analysis1[2*(r-4), 0] = np.mean(count)
    analysis1[2*(r-4), 1] = np.var(count)

Glo_points = np.loadtxt('Glocoordinates.txt')
Grc_points = np.loadtxt('Grccoordinates.txt')
MF_GL = np.loadtxt('MF_GL.txt')
GRC_GL = np.loadtxt('GRC_GL.txt')

total = np.vstack((Glo_points, Grc_points))
tree_total = spatial.cKDTree(total)

####Searching Glomeruli near grc
rs = np.arange(4, 8, 0.5)
analysis = np.empty(shape=(np.size(rs), 2))
parallel(n_jobs=8)(delyaed(count_neighborhood_grc)(Grc_points, tree_total, r) for r in rs)

####Searching Grc near Glo
analysis1 = np.empty(shape=(np.size(rs), 2))
parallel(n_jobs=8)(delyaed(count_neighborhood_glo)(Glo_points, tree_total, r) for r in rs)

np.savetxt('GlonearGrc', analysis)
np.savetxt('GrcnearGlo', analysis1)