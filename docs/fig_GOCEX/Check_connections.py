# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.2'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Parallel fiber - GoC

# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib widget

# %%
output_path =  Path('/Users/shhong/Dropbox/network_data/model_pybrep_output/checked')

# %%
adend = np.loadtxt(output_path / 'GoCadendcoordinates.dat')
bdend = np.loadtxt(output_path / 'GoCbdendcoordinates.dat')
gocs = np.loadtxt(output_path / "GoCcoordinates.sorted.dat")

# %%
adend = adend.reshape((1995,50,3))
bdend = bdend.reshape((1995,24,3))

# %%
gocx = gocs[:, 0]
gocy = gocs[:, 1]


# %%
ii = (gocx<600)*(gocx>300)*(gocy>300)*(gocy<600)
ii, = np.where(ii)
ii = np.random.choice(ii, 4)
ii

# %% [markdown]
# Here we check if the dendritic point alignment is ok.

# %%
draw_dend = lambda ax, x, y, z, c: ax.plot(x, y, z, '-', linewidth=2.5, c=c)
draw_apic = lambda ax, x, y, z: draw_dend(ax, x, y, z, 'c')
draw_basal = lambda ax, x, y, z: draw_dend(ax, x, y, z, 'm')
draw_dendpts = lambda ax, x, y, z, s: ax.plot(x, y, z, '.k', markersize=s)
draw_apicpts = lambda ax, x, y, z: draw_dendpts(ax, x, y, z, 2)
draw_basalpts = lambda ax, x, y, z: draw_dendpts(ax, x, y, z, 1)


# %%
plt.close('all')
fig = plt.figure(figsize=(8.5/2.54, 8.5/2.54))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=-50, elev=35)

def draw1(ax, adend, bdend, i):
       x, y, z = adend[i][:,0], adend[i][:,1], adend[i][:,2]
       draw_apic(ax, x, y, z)
       x, y, z = bdend[i][:,0], bdend[i][:,1], bdend[i][:,2]
       draw_basal(ax, x, y, z)

def draw1pts(ax, adend, bdend, i):
       x, y, z = adend[i][:,0], adend[i][:,1], adend[i][:,2]
       draw_apicpts(ax, x, y, z)
       x, y, z = bdend[i][:,0], bdend[i][:,1], bdend[i][:,2]
       draw_basalpts(ax, x, y, z)

for i in ii:
       draw1(ax, adend, bdend, i)
       draw1pts(ax, adend, bdend, i)
    
ax.set(xlabel='x (μm)', ylabel='y (μm)', zlabel='z (μm)',
       xlim=[300, 600], ylim=[350, 650], zlim=[0, 450],
       xticks=np.arange(300, 601, 100), yticks=np.arange(0, 301, 100)+350)
plt.savefig('goc_cloud.jpg', dpi=300)
plt.savefig('goc_cloud.pdf', dpi=300)


# %% [markdown]
# Now we check if connections look ok.

# %%
grcs = np.loadtxt(output_path / "GCcoordinates.sorted.dat")
gcts = np.loadtxt(output_path / "GCTcoordinates.sorted.dat")

# %%
data_index = 12

data_dir = Path('/Users/shhong/Dropbox/network_data/model_pybrep_output/')
fcoords = "PFtoGoCcoords{}.dat".format(data_index)
fsrcs = "PFtoGoCsources{}.dat".format(data_index)
ftgts = "PFtoGoCtargets{}.dat".format(data_index)
fdsts = "PFtoGoCdistances{}.dat".format(data_index)


xyz = np.loadtxt(data_dir / fcoords)
srcs = np.loadtxt(data_dir / fsrcs).astype(int)
tgts = np.loadtxt(data_dir / ftgts).astype(int)
dsts = np.loadtxt(data_dir / fdsts)

# %%
i = 800
tgt = tgts[i]
src = srcs[i]
dst = dsts[i]

pt2 = xyz[i]

print("{} -> {} with d = {}".format(src, tgt, dst))

# %%
plt.close('all')

fig = plt.figure(figsize=(8.5/2.54*8/5*0.8, 8.5/2.54*0.8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=-50, elev=35)

pt0 = grcs[src]
pt1 = gcts[src]
ax.plot([pt0[0]], [pt0[1]], [pt0[2]], 'ok', markersize=4)
ax.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], [pt0[2], pt1[2]], '-k')
ax.plot([pt1[0], pt1[0]+1e3], [pt1[1], pt1[1]], [pt1[2], pt1[2]], '-k')
ax.plot([pt1[0], pt1[0]-1e3], [pt1[1], pt1[1]], [pt1[2], pt1[2]], '-k')
draw1(ax, adend, bdend, tgt)
draw1pts(ax, adend, bdend, tgt)


pt2 = xyz[i]
ax.plot([pt2[0]], [pt2[1]], [pt2[2]], 'or', markersize=4)
ax.set(xlabel='x (μm)', ylabel='y (μm)', zlabel='z (μm)',
       xlim=[300, 700], ylim=[150, 250], zlim=[0, 450],
       xticks=np.arange(300, 701, 100), yticks=np.arange(150, 251, 50), zticks=[0, 200, 400])

plt.savefig('goc_pf.jpg', dpi=300)
plt.savefig('goc_pf.pdf', dpi=300)




# %%
fig, ax = plt.subplots(figsize=(8.5/2.54*8/5*0.7, 8.5/2.54*0.7))


ax.plot([pt0[0], pt1[0]], [pt0[2], pt1[2]], '-k')
ax.plot([pt0[0]], [pt0[2]], 'o-k', markersize=4)
ax.plot([pt1[0], pt1[0]+1e3], [pt1[2], pt1[2]], '-k')
ax.plot([pt1[0], pt1[0]-1e3], [pt1[2], pt1[2]], '-k')
ax.plot(adend[tgt][:,0], adend[tgt][:,2],'-c', linewidth=2.5)
ax.plot(bdend[tgt][:,0], bdend[tgt][:,2],'-m', linewidth=2.5)
ax.plot(adend[tgt][:,0], adend[tgt][:,2],'.k', markersize=2)
ax.plot(bdend[tgt][:,0], bdend[tgt][:,2],'.k', markersize=1)
ax.plot([pt2[0]], [pt2[2]], 'or')
ax.set(xlabel='x (μm)', ylabel='z (μm)',
       xlim=[250, 700], ylim=[0, 450])
plt.tight_layout()
plt.savefig('goc_pf_p.jpg', dpi=300)



# %% [markdown]
# ## Ascending axon - GoC

# %%
data_index = 80

data_dir = output_path
fcoords = "AAtoGoCcoords{}.dat".format(data_index)
fsrcs = "AAtoGoCsources{}.dat".format(data_index)
ftgts = "AAtoGoCtargets{}.dat".format(data_index)
fdsts = "AAtoGoCdistances{}.dat".format(data_index)

xyz = np.loadtxt(data_dir / fcoords)
srcs = np.loadtxt(data_dir / fsrcs).astype(int)
tgts = np.loadtxt(data_dir / ftgts).astype(int)
dsts = np.loadtxt(data_dir / fdsts)

# %%
i = 6750
tgt = tgts[i]
src = srcs[i]
dst = dsts[i]

pt2 = xyz[i]

print("{} -> {} with d = {}".format(src, tgt, dst))

# %%
plt.close('all')

fig = plt.figure(figsize=(8.5/2.54*8/5*0.8, 8.5/2.54*0.8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=-18, elev=10)

pt0 = grcs[src]
pt1 = gcts[src]
ax.plot([pt0[0]], [pt0[1]], [pt0[2]], 'ok', markersize=4)
ax.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], [pt0[2], pt1[2]], '-k')
ax.plot([pt1[0], pt1[0]+1e3], [pt1[1], pt1[1]], [pt1[2], pt1[2]], '-k')
ax.plot([pt1[0], pt1[0]-1e3], [pt1[1], pt1[1]], [pt1[2], pt1[2]], '-k')
draw1(ax, adend, bdend, tgt)
draw1pts(ax, adend, bdend, tgt)


pt2 = xyz[i]
ax.plot([pt2[0]], [pt2[1]], [pt2[2]], 'or', markersize=4)
ax.set(xlabel='x (μm)', ylabel='y (μm)', zlabel='z (μm)',
       xlim=[850, 950], ylim=[375, 575], zlim=[0, 450],
       xticks=np.arange(900, 901, 100), yticks=np.arange(400, 551, 50), zticks=[0, 200, 400])

plt.savefig('goc_aa.jpg', dpi=300)
plt.savefig('goc_aa.pdf', dpi=300)




# %%
fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=True)
ax[0].plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 'o-k')
ax[0].plot([pt1[0], pt1[0]+1e3], [pt1[1], pt1[1]], '-k')
ax[0].plot([pt1[0], pt1[0]-1e3], [pt1[1], pt1[1]], '-k')
ax[0].plot(adend[tgt][:,0], adend[tgt][:,1], '.r')
ax[0].plot(bdend[tgt][:,0], bdend[tgt][:,1], '.m')
ax[0].plot([pt2[0]], [pt2[1]], 'oc')

ax[1].plot([pt0[1], pt1[1]], [pt0[2], pt1[2]], 'o-k')
ax[1].plot([pt1[1], pt1[1]+1e3], [pt1[2], pt1[2]], '-k')
ax[1].plot([pt1[1], pt1[1]-1e3], [pt1[2], pt1[2]], '-k')
ax[1].plot(adend[tgt][:,1], adend[tgt][:,2],'.r')
ax[1].plot(bdend[tgt][:,1], bdend[tgt][:,2],'.m')
ax[1].plot([pt2[1]], [pt2[2]], 'oc')
ax[1].set(xlim=[300, 550])

# %%
fig, ax = plt.subplots(figsize=(8.5/2.54*8/5*0.7, 8.5/2.54*0.7))


ax.plot([pt0[1], pt1[1]], [pt0[2], pt1[2]], '-k')
ax.plot([pt0[1]], [pt0[2]], 'o-k', markersize=4)

ax.plot([pt1[1], pt1[1]+1e3], [pt1[2], pt1[2]], '-k')
ax.plot([pt1[1], pt1[1]-1e3], [pt1[2], pt1[2]], '-k')
ax.plot(adend[tgt][:,1], adend[tgt][:,2],'-c', linewidth=2.5)
ax.plot(bdend[tgt][:,1], bdend[tgt][:,2],'-m', linewidth=2.5)
ax.plot(adend[tgt][:,1], adend[tgt][:,2],'.k', markersize=2)
ax.plot(bdend[tgt][:,1], bdend[tgt][:,2],'.k', markersize=1)
ax.plot([pt2[1]], [pt2[2]], 'or')
ax.set(xlabel='y (μm)', ylabel='z (μm)',
       xlim=[300, 550], ylim=[0, 450])
plt.tight_layout()
plt.savefig('goc_aa_p.jpg', dpi=300)


# %% [markdown]
# ## Convert big files to the npy format

# %%
output_path = Path('/Users/shhong/Dropbox/network_data/output_ines_2')
# output_path = Path('/Users/shhong/Dropbox/network_data/output_brep')
src = np.loadtxt(output_path / "PFtoGoCsources.dat")
tgt = np.loadtxt(output_path / "PFtoGoCtargets.dat")
src = src.astype(int)
tgt = tgt.astype(int)

np.save(output_path / 'PFtoGoCsources.npy', src)
np.save(output_path / 'PFtoGoCtargets.npy', tgt)

# %%
src = np.loadtxt(output_path / "AAtoGoCsources.dat")
tgt = np.loadtxt(output_path / "AAtoGoCtargets.dat")
src = src.astype(int)
tgt = tgt.astype(int)

np.save(output_path / 'AAtoGoCsources.npy', src)
np.save(output_path / 'AAtoGoCtargets.npy', tgt)

# %% [markdown]
# ## Load back data and check PFs

# %%
output_path = Path('/Users/shhong/Dropbox/network_data/output_ines_2')

srcs = np.load(output_path / 'PFtoGoCsources.npy')
tgts = np.load(output_path / 'PFtoGoCtargets.npy')

grcxy = np.loadtxt(output_path / 'GCcoordinates.sorted.dat')
gocxy = np.loadtxt(output_path / 'GoCcoordinates.sorted.dat')

# %%
import dask.dataframe as dd

df = dd.from_array(np.vstack((srcs, tgts)).T, columns=('src', 'tgt'))

# %%
cons_per_goc = df.groupby('tgt').count().compute()
cons_per_goc
cons_per_pf = df.groupby('src').count().compute()
cons_per_pf

# %%
import dask.dataframe as dd

def convert_from_dd(x, size):
    temp = np.zeros(size)
    temp[x.index] = x.values
    return temp

cons_per_goc = convert_from_dd(cons_per_goc.src, gocxy.shape[0])
cons_per_pf = convert_from_dd(cons_per_pf.tgt, grcxy.shape[0])

# %%
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
ax[0,0].plot(cons_per_goc, '.')
ax[0,0].set(xlabel="GoC id", ylabel='Connections')
_ = ax[0,1].hist(cons_per_goc, 200)
ax[0,1].set(xlabel="Connections per GoC", ylabel='Count')
ax[1,0].scatter(gocxy[:,0], gocxy[:,1], 100, cons_per_goc, '.')
ax[1,0].set(xlabel="x (um)", ylabel="y (um)")
ax[1,1].scatter(gocxy[:,1], gocxy[:,2], 100, cons_per_goc, '.')
ax[1,1].set(xlabel="x (um)", ylabel="z (um)")

# %%
print("Connections per GoC = {} ± {}".format(np.mean(cons_per_goc), np.std(cons_per_goc)/np.sqrt(cons_per_goc.size)))

# %%
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
ax[0,0].plot(cons_per_pf, '.')
ax[0,0].set(xlabel="PF id", ylabel='Connections')
ax[0,1].set(xlabel="Connections per PF", ylabel='Count')
_ = ax[0,1].hist(cons_per_pf, 200)
ax[1,0].scatter(grcxy[:,0], grcxy[:,1], 0.25, cons_per_pf, '.')
ax[1,0].set(xlabel="x (um)", ylabel="y (um)")
ax[1,1].scatter(grcxy[:,1], grcxy[:,2], 0.25, cons_per_pf, '.')
ax[1,1].set(xlabel="x (um)", ylabel="z (um)")

# %%
print("Connections per PF = {} ± {}".format(np.mean(cons_per_pf), np.std(cons_per_pf)/np.sqrt(cons_per_pf.size)))

# %% [markdown]
# ## Perform the same analysis on BREP outputs

# %%
# output_path = Path('/Users/shhong/Dropbox/network_data/output_brep')
# src = np.loadtxt(output_path / "PFtoGoCsources.dat")
# tgt = np.loadtxt(output_path / "PFtoGoCtargets.dat")
# src = src.astype(int)
# tgt = tgt.astype(int)

# np.save(output_path / 'PFtoGoCsources.npy', src)
# np.save(output_path / 'PFtoGoCtargets.npy', tgt)

# src = np.loadtxt(output_path / "AAtoGoCsources.dat")
# tgt = np.loadtxt(output_path / "AAtoGoCtargets.dat")
# src = src.astype(int)
# tgt = tgt.astype(int)

# np.save(output_path / 'AAtoGoCsources.npy', src)
# np.save(output_path / 'AAtoGoCtargets.npy', tgt)

# %%
output_path = Path('/Users/shhong/Dropbox/network_data/output_brep')

srcs = np.load(output_path / 'PFtoGoCsources.npy')
tgts = np.load(output_path / 'PFtoGoCtargets.npy')
grcxy = np.loadtxt(output_path / 'GCcoordinates.sorted.dat')
gocxy = np.loadtxt(output_path / 'GoCcoordinates.sorted.dat')

df = dd.from_array(np.vstack((srcs, tgts)).T, columns=('src', 'tgt'))

# %%
cons_per_goc = df.groupby('tgt').count().compute()
cons_per_pf = df.groupby('src').count().compute()

cons_per_goc = convert_from_dd(cons_per_goc.src, gocxy.shape[0])
cons_per_pf = convert_from_dd(cons_per_pf.tgt, grcxy.shape[0])

# %%
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
ax[0,0].plot(cons_per_goc, '.')
ax[0,0].set(xlabel="GoC id", ylabel='Connections')
_ = ax[0,1].hist(cons_per_goc, 200)
ax[0,1].set(xlabel="Connections per GoC", ylabel='Count')
ax[1,0].scatter(gocxy[:,0], gocxy[:,1], 100, cons_per_goc, '.')
ax[1,0].set(xlabel="x (um)", ylabel="y (um)")
ax[1,1].scatter(gocxy[:,1], gocxy[:,2], 100, cons_per_goc, '.')
ax[1,1].set(xlabel="x (um)", ylabel="z (um)")

# %%
print("Connections per GoC = {} ± {}".format(np.mean(cons_per_goc), np.std(cons_per_goc)/np.sqrt(cons_per_goc.size)))

# %%
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
ax[0,0].plot(cons_per_pf, '.')
ax[0,0].set(xlabel="PF id", ylabel='Connections')
ax[0,1].set(xlabel="Connections per PF", ylabel='Count')
_ = ax[0,1].hist(cons_per_pf, 200)
ax[1,0].scatter(grcxy[:,0], grcxy[:,1], 0.25, cons_per_pf, '.')
ax[1,0].set(xlabel="x (um)", ylabel="y (um)")
ax[1,1].scatter(grcxy[:,1], grcxy[:,2], 0.25, cons_per_pf, '.')
ax[1,1].set(xlabel="x (um)", ylabel="z (um)")

# %%
print("Connections per PF = {} ± {}".format(np.mean(cons_per_pf), np.std(cons_per_pf)/np.sqrt(cons_per_pf.size)))

# %%
9708.829573934838/7840.988471177945

# %%
24.272073934837092/19.602471177944864

# %%
np.sqrt(1.238215004)

# %%
