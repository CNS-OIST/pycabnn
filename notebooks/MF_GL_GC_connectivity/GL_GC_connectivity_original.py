# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Statistical analysis of the generated cell positions
#
# This notebook contains some codes that analyze the cell position data generated from pycabnn.

# +
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm, trange
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors


# + [markdown] tags=[]
# ## Load data
#
# Here we load the sample cell position data.

# +
# A few utility functions to p

def limit_to_box(x, box):
    """select only the points within a given box."""
    mf = x.copy()
    for i, t in enumerate(box):
        mf = mf[mf[:, i] >= t[0], :]
        mf = mf[mf[:, i] <= t[1], :]
    return mf

def print_range(goc):
    """prints the 3d range occupied by the cell population"""
    print(
        "Current range:\n",
        "x: [{}, {}]\n".format(goc[:, 0].min(), goc[:, 0].max()),
        "y: [{}, {}]\n".format(goc[:, 1].min(), goc[:, 1].max()),
        "z: [{}, {}]".format(goc[:, 2].min(), goc[:, 2].max()),
    )

def fix_coords(x, bbox):
    """removes the cells in a 25 um-wide gutter."""
    y = x - 25
    y = limit_to_box(y, bbox)
    print_range(y)
    return y



# +
# load the data
fname = "../../test_data/generated_positions/coords_20190626_1_6.npz"
f = np.load(fname)
f['grc_nop'].shape # grc_nop is grc positions without perturbation

# readjuct the data to the bounding box - note that the data were generated in a larger volume
bbox = [[0, 700], [0, 700], [0, 200]]
grc = fix_coords(f['grc_nop'], bbox) # here we use the unperturbed position
glo = fix_coords(f['glo'], bbox)
grx = grc + np.random.randn(*grc.shape)*0.2 # add perturbation, corresponding to the softness margin, 0.2 um here.

# -

# ## Simple distance-based GrC-Glo connection
#
# Here we demonstrate and analyze the GrC-glomerulus connectivity based on the simple distance-based rule.

# ## Pycabnn-generated cell position

# +
## [0,1,2] = [mediolateral, sagittal, z]
## we first rescale the sagittal direction to make the connection stretch more in the sagittal direction

scale_factor = 1/4

src = glo.copy() #GL
tgt = grx.copy() #GrC

src[:, 1] *= scale_factor
tgt[:, 1] *= scale_factor
# -

rr = 7.85
nn = NearestNeighbors()
nn.fit(src)
conns = nn.radius_neighbors(tgt, radius=rr, return_distance=False)
print("connection list = ", conns)

# Check the number of GrC is the same as the length of the connection list
assert tgt.shape[0] == conns.shape[0]


# +
# Transform conns into the pandas table

def transform_conns_table(conns):
    Nconns = sum(c.size for c in conns)

    src = np.zeros(Nconns, dtype='int')
    tgt = np.zeros(Nconns, dtype='int')

    count = 0
    for i in trange(conns.size):
        n = conns[i].size
        tgt[count:(count+n)] = i
        src[count:(count+n)] = conns[i]
        count +=n

    return pd.DataFrame({'src':src, 'tgt':tgt})

df_conns = transform_conns_table(conns)
df_conns


# -

# ### Now we analyze the connectivityhist

# +
def plot_conns(df_conns):
    nconns_per_grc = df_conns.groupby('tgt').count().values.ravel()
    nconns_per_gl = df_conns.groupby('src').count().values.ravel()

    print('N conns per GrC = {} ± {}'.format(np.mean(nconns_per_grc), np.std(nconns_per_grc)))
    print('N conns per GL = {} ± {}'.format(np.mean(nconns_per_gl), np.std(nconns_per_gl)))

    _, (ax1, ax2) = plt.subplots(ncols=2)
    n_r, x_r, _ = ax1.hist(nconns_per_grc, np.arange(nconns_per_grc.max()),10)
    ax1.set_title('count', loc="left")
    ax1.set(xlabel='connections per GrC')

    _ = ax2.hist(nconns_per_gl, np.arange(nconns_per_gl.max()),10)
    ax2.set(xlabel='connections per GL')

plot_conns(df_conns)
# -

# ### Dendrite length

# +
src = glo.copy() #GL
tgt = grx.copy() #GrC

src_coord = src[df_conns.src.values.ravel(),:]
tgt_coord = tgt[df_conns.tgt.values.ravel(),:]
# -

print(tgt.shape)
print(df_conns.tgt.values.ravel().max()

delta = tgt_coord-src_coord
dend_length = np.sqrt((delta**2).sum(axis=1))

plt.hist(dend_length)

# ## Comparison with random cell positions
#
# The same connectivity analysis with the purely random cell positions. The result can be slightly different from the numbers that we reported in the paper --- since it is random!

# +
# generate the purely rarndom cell position
grx = np.random.rand(*grx.shape)
glo = np.random.rand(*glo.shape)

grx[:,0] *= 700
grx[:,1] *= 700
grx[:,2] *= 200

glo[:,0] *= 700
glo[:,1] *= 700
glo[:,2] *= 200

scale_factor = 1/4

# grx = grc + np.random.randn(*grc.shape)*0.2

src = grx.copy()
tgt = glo.copy()
src[:, 1] *= scale_factor
tgt[:, 1] *= scale_factor
# -

nn = NearestNeighbors()
nn.fit(tgt)
df_rand = transform_conns_table(nn.radius_neighbors(src, radius=rr, return_distance=False))

plot_conns(df_rand)


