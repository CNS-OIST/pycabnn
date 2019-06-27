# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Glomerulus - GrC connectivity
import numpy as np
import matplotlib.pyplot as plt

fname = "../fig_POPGEN/coords_20190626_1_4.npz"
f = np.load(fname)
f['grc_nop'].shape


def limit_to_box(x, box):
    mf = x.copy()
    for i, t in enumerate(box):
        mf = mf[mf[:, i] >= t[0], :]
        mf = mf[mf[:, i] <= t[1], :]
    return mf


def print_range(goc):
    print(
        "Current range:\n",
        "x: [{}, {}]\n".format(goc[:, 0].min(), goc[:, 0].max()),
        "y: [{}, {}]\n".format(goc[:, 1].min(), goc[:, 1].max()),
        "z: [{}, {}]".format(goc[:, 2].min(), goc[:, 2].max()),
    )


def fix_coords(x, bbox):
    y = x - 25
    y = limit_to_box(y, bbox)
    print_range(y)
    return y


bbox = [[0, 700], [0, 700], [0, 200]]
grc = fix_coords(f['grc_nop'], bbox)
glo = fix_coords(f['glo'], bbox)

scale_factor = 1/5.5

src = grx.copy()
tgt = glo.copy()
src[:, 1] *= scale_factor
tgt[:, 1] *= scale_factor

from sklearn.neighbors import NearestNeighbors

# +
nn = NearestNeighbors()
nn.fit(tgt)
# conns = nn.radius_neighbors(src, radius=7, return_distance=False)
# nconns = np.frompyfunc(lambda x: x.size, 1, 1)(conns).astype(int)
# _ = plt.hist(nconns,np.arange(nconns.max()),100)
# print('Mean connection = {}'.format(np.mean(nconns)))
      
conns = nn.kneighbors(src, n_neighbors=4, return_distance=False)
# -

dendvs = np.vstack([glo[conn,:] - grc[i,:] for i, conn in enumerate(conns) if conn.size>1])
dendlens = np.sqrt((dendvs**2).sum(axis=-1))
dendlens

plt.hist(dendlens,500)
plt.xlim([7.5, 40])
print('{}±{}'.format(dendlens.mean(), dendlens.std()))

# +
dendvs = [glo[conns[i],:] - grc[i,:] for i, conn in enumerate(conns) if conn.size>1]

ml_spread = np.array([z[:,0].max()-z[:,0].min() for z in dendvs])

plt.hist(ml_spread,100)
print('{}±{}'.format(ml_spread.mean(), ml_spread.std()))

sg_spread = np.array([z[:,1].max()-z[:,1].min() for z in dendvs])

plt.hist(sg_spread,100)
print('{}±{}'.format(sg_spread.mean(), sg_spread.std()))

# +
grx = grc + np.random.randn(*grc.shape)*0.25
nn.fit(grx)
dists, nnids = nn.kneighbors(grx, n_neighbors=2, return_distance=True)

nn.fit(grc)
dists_u, nnids = nn.kneighbors(grc, n_neighbors=2, return_distance=True)

nnids = nnids[:,1]
dists = dists[:,1]

dists_u = dists_u[:,1]
# -

_ = plt.hist(dists, 500)
# _ = plt.hist(dists_u, 500)
plt.xlim([4, 10])

# +
gry = limit_to_box(grx, [[30, 670], [30, 670], [30, 170]])

nn = NearestNeighbors(n_jobs=-1)
nn.fit(grx)

from tqdm.autonotebook import tqdm

mcounts = []
sdcounts = []
dists = np.linspace(0, 30, 120)
for r in tqdm(dists):
    count = np.frompyfunc(lambda x: x.size, 1, 1)(nn.radius_neighbors(
        gry, radius=r, return_distance=False
    )).astype(float) - 1
    mcounts.append(count.mean())
    sdcounts.append(count.std()/np.sqrt(count.size))
# mcount = count.mean()
# sdcount = count.std()
# print('{} ± {}'.format(mcount, sdcount))

cc2 = np.gradient(mcounts)/(dists**2)
cc2_0 = cc2[-1]
cc2 = cc2/cc2_0
plt.plot(dists, cc2)

mcounts = np.array(mcounts)
sdcounts = np.array(sdcounts)

_, ax = plt.
cc2_u = np.gradient(mcounts + 150*sdcounts)/(dists**2+0.001)/cc2_0
cc2_d = np.gradient(mcounts - 150*sdcounts)/(dists**2+0.001)/cc2_0
# plt.fill_between(dists, cc2_d, cc2_u)
plt.fill_between(dists, cc2_d, cc2_u, alpha=0.5)
plt.plot(dists, cc2, 'k')
# -

scale_factor = 1/3
gry = limit_to_box(grx, [[40, 660], [40/scale_factor, 700-40/scale_factor], [0, 200]])
src = gry.copy()
tgt = glo.copy()
src[:, 1] *= scale_factor
tgt[:, 1] *= scale_factor

# +
nn = NearestNeighbors(n_jobs=-1)
nn.fit(tgt)
# conns = nn.radius_neighbors(src, radius=7, return_distance=False)
# nconns = np.frompyfunc(lambda x: x.size, 1, 1)(conns).astype(int)
# _ = plt.hist(nconns,np.arange(nconns.max()),100)
# print('Mean connection = {}'.format(np.mean(nconns)))

mcounts = []
vcounts = []
dists0 = np.linspace(0, 40, 120)
for r in tqdm(dists0):
    count = np.frompyfunc(lambda x: x.size, 1, 1)(nn.radius_neighbors(
        src, radius=r, return_distance=False
    )).astype(float)
    mcounts.append(count.mean())
    vcounts.append(count.var())
# -

mcounts0 = np.array(mcounts)
vcounts0 = np.array(vcounts)

# +
dists0 = np.linspace(0, 40, 120)


ii = (mcounts0>0) * (mcounts0 < 80)
mcounts = mcounts0[ii]
dists = dists0[ii]
vcounts = vcounts0[ii]
# -

plt.plot(mcounts, (vcounts)/mcounts, mcounts, (mcounts/86)**(4/3)+0.36)
# plt.xlim([0, 20])

plt.plot(dists, (vcounts)/mcounts)
# plt.xlim([0, 20])

# +
plt.plot(dists, mcounts, 'o-')
plt.ylim([0, 15])


# -


