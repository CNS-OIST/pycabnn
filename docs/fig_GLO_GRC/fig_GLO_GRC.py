# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Glomerulus - GrC connectivity
import numpy as np
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

fname = "../fig_POPGEN/data/coords_20190626_1_6.npz"
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
grx = grc + np.random.randn(*grc.shape)*0.2

# +
grx = np.random.rand(*grx.shape)
glo = np.random.rand(*glo.shape)

grx[:,0] *= 700
grx[:,1] *= 700
grx[:,2] *= 200

glo[:,0] *= 700
glo[:,1] *= 700
glo[:,2] *= 200

# +
scale_factor = 1/4

# grx = grc + np.random.randn(*grc.shape)*0.2

src = grx.copy()
tgt = glo.copy()
src[:, 1] *= scale_factor
tgt[:, 1] *= scale_factor
# -

from sklearn.neighbors import NearestNeighbors

# +
rr = 7.85
nn = NearestNeighbors()
nn.fit(tgt)
conns = nn.radius_neighbors(src, radius=rr, return_distance=False)
nconns = np.frompyfunc(lambda x: x.size, 1, 1)(conns).astype(int)

n_r, x_r, _ = plt.hist(nconns,np.arange(nconns.max()),100)
print('N conns = {} ± {}'.format(np.mean(nconns), np.std(nconns)))
      
# conns = nn.kneighbors(src, n_neighbors=2, return_distance=False)

# +
nn = NearestNeighbors()
nn.fit(tgt)
conns = nn.radius_neighbors(src, radius=rr, return_distance=False)
nconns = np.frompyfunc(lambda x: x.size, 1, 1)(conns).astype(int)
nc, xc, _ = plt.hist(nconns, np.arange(nconns.max()), 100)
print('N conns = {} ± {}'.format(np.mean(nconns), np.std(nconns)))
      
# conns = nn.kneighbors(src, n_neighbors=2, return_distance=False)
# -

plt.bar(x_r[:-1], n_r, 1, alpha=0.5)
plt.bar(xc[:-1], nc, 1, alpha=0.5)

1.364489593641605/2.126870358595118

dendvs = np.vstack([glo[conn,:] - grc[i,:] for i, conn in enumerate(conns) if conn.size>1])
dendlens = np.sqrt((dendvs**2).sum(axis=-1))
dendlens

plt.hist(dendlens,500)
plt.xlim([0, 40])
plt.ylim([0, 5000])
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
nn.fit(grx)
dists, nnids = nn.kneighbors(grx, n_neighbors=2, return_distance=True)

nn.fit(grc)
dists_u, nnids = nn.kneighbors(grc, n_neighbors=2, return_distance=True)

nnids = nnids[:,1]
dists = dists[:,1]

dists_u = dists_u[:,1]
# -

_, ax = plt.subplots(figsize=(8.9/2.54, 8.9/2.54*5/8))
ax.hist(dists, 450, density=True, color='k')
# _ = plt.hist(dists_u, 500)
ax.set(
    xlim=[5, 10],
    xlabel='nearest neighbor distance (μm)',
    ylabel='density'
)
plt.tight_layout()
plt.savefig('nn_dist_hist.png', dpi=600)

# +
gry = limit_to_box(grx, [[30, 670], [30, 670], [30, 170]])

nn = NearestNeighbors(n_jobs=-1)
nn.fit(grx)

from tqdm.autonotebook import tqdm

mcounts = []
sdcounts = []
dists = np.linspace(0, 30, 240)
for r in tqdm(dists):
    count = np.frompyfunc(lambda x: x.size, 1, 1)(nn.radius_neighbors(
        gry, radius=r, return_distance=False
    )).astype(float) - 1
    mcounts.append(count.mean())
    sdcounts.append(count.std()/np.sqrt(count.size))
# mcount = count.mean()
# sdcount = count.std()
# print('{} ± {}'.format(mcount, sdcount))

# +
cc2 = np.gradient(mcounts)/(dists**2)
cc2_0 = cc2[-1]
cc2 = cc2/cc2_0

mcounts = np.array(mcounts)
sdcounts = np.array(sdcounts)

cc2_u = np.gradient(mcounts + 150*sdcounts)/(dists**2+0.001)/cc2_0
cc2_d = np.gradient(mcounts - 150*sdcounts)/(dists**2+0.001)/cc2_0
# plt.fill_between(dists, cc2_d, cc2_u)
# plt.fill_between(dists, cc2_d, cc2_u, alpha=0.5)


_, ax = plt.subplots(figsize=(8.9/2.54, 8.9/2.54*5/8))
ax.plot(dists, cc2, 'k')
ax.set(
    ylim = [0, 3],
    xlim = [0, 30],
    xlabel='radial distance (μm)',
    ylabel='pair correlation function'
)
plt.tight_layout()
plt.savefig('cc2_grc.png', dpi=600)
# -



scale_factor = 1/4
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
dists0 = np.linspace(1, 40, 120)
for r in tqdm(dists0):
    count = np.frompyfunc(lambda x: x.size, 1, 1)(nn.radius_neighbors(
        src, radius=r, return_distance=False
    )).astype(float)
    mcounts.append(count.mean())
    vcounts.append(count.var())
# -

mcounts0 = np.array(mcounts)
vcounts0 = np.array(vcounts)

ii = (mcounts0>0) * (mcounts0 < 60)
mcounts = mcounts0[ii]
dists = dists0[ii]
vcounts = vcounts0[ii]

mcr = np.interp(dists, dists_r, ((vcounts_r)/mcounts_r))

# +
 _, ax = plt.subplots(figsize=(8.9/2.54, 8.9/2.54*5/8))
#_, ax = plt.subplots()

# ax.plot(
#      mcounts, ((vcounts)/mcounts)/mcr, 'k',
#      mcounts, mcr/mcr, '--g'
# )


ax.plot(
    2.128*dists, (((vcounts)/mcounts)/mcr), 'k',
    2.128*dists, mcr/mcr, '--g'
)
ax.plot(np.array([20.26, 20.26]), [0.31, 0.5], ':b')
ax.annotate('6.1 glomeruli', [20.-1, 0.52], color='b')
ax.plot(np.array([17.47, 17.47]), [0.31, 0.63], ':r')
ax.annotate('4 glomeruli', [17.-3, 0.65], color='r')
ax.annotate('50 glomeruli', [32., 0.75], color='c')
ax.plot(np.array([42., 42.]), [0.31, 0.72], ':c')

ax.annotate('random', [37, 0.93], color='g')

ax.set(
#     xticks=np.arange(0, 44, 4),
    xlabel='average search range (μm)',
    ylabel='Fano factor (normalized)'
)

plt.tight_layout()
plt.savefig('var_conns.png', dpi=300)

# plt.xlim([0, 60])

# plt.ylim([-0.02, 0.02])

# +
 _, ax = plt.subplots(figsize=(8.9/2.54, 8.9/2.54*5/8))
#_, ax = plt.subplots()

# ax.plot(
#      mcounts, ((vcounts)/mcounts)/mcr, 'k',
#      mcounts, mcr/mcr, '--g'
# )


ax.semilogy(
    2.128*dists, mcounts, 'k'
)



# +
print(2.128*dists[(((vcounts)/mcounts)/mcr).argmin()])
print(mcounts[(((vcounts)/mcounts)/mcr).argmin()])
print(mcounts[abs(dists-42/2.128).argmin()])
# print(mcounts[abs(dists-40).argmin()])



# -

plt.semilogy(mcounts, (vcounts)/mcounts, 'k',
         mcounts, (mcounts/86)**(4/3)+0.36)
# plt.xlim([0, 80])

plt.plot(mcounts, (vcounts)/mcounts, 'k',
         mcounts, (mcounts/86)**(4/3)+0.36)
plt.xlim([0, 20])
plt.ylim([0.3, 0.5])

plt.plot(dists, (vcounts)/mcounts)
# plt.xlim([0, 20])

plt.plot(dists, mcounts, 'o-')
plt.ylim([0, 15])

mcounts_r = mcounts.copy()
vcounts_r = vcounts.copy()
dists_r = dists.copy()

plt.loglog(mcounts_r, (vcounts_r)/mcounts_r)


