
# coding: utf-8

# ## Parallel fiber - GoC

# In[8]:


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

get_ipython().run_line_magic('matplotlib', 'nbagg')


# In[2]:


output_path = Path.cwd().parent.parent / 'output_2'


# In[3]:


adend = np.loadtxt(output_path / 'GoCadendcoordinates.dat')
bdend = np.loadtxt(output_path / 'GoCbdendcoordinates.dat')
grcs = np.loadtxt(output_path / "GCcoordinates.sorted.dat")
gcts = np.loadtxt(output_path / "GCTcoordinates.dat")


# In[4]:


adend = adend.reshape((1995,50,3))
bdend = bdend.reshape((1995,24,3))


# In[ ]:


i = 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(adend[i][:,0], adend[i][:,1], adend[i][:,2],'.r')
ax.plot(bdend[i][:,0], bdend[i][:,1], bdend[i][:,2],'.k')

i = 1000
ax.plot(adend[i][:,0], adend[i][:,1], adend[i][:,2],'.r')
ax.plot(bdend[i][:,0], bdend[i][:,1], bdend[i][:,2],'.k')


# In[42]:


data_index = 10

data_dir = output_path
fcoords = "PFtoGoCcoords{}.dat".format(data_index)
fsrcs = "PFtoGoCsources{}.dat".format(data_index)
ftgts = "PFtoGoCtargets{}.dat".format(data_index)
fdsts = "PFtoGoCdistances{}.dat".format(data_index)


xyz = np.loadtxt(data_dir / fcoords)
srcs = np.loadtxt(data_dir / fsrcs).astype(int)
tgts = np.loadtxt(data_dir / ftgts).astype(int)
dsts = np.loadtxt(data_dir / fdsts)


# In[43]:


i = 4000
tgt = tgts[i]
src = srcs[i]
dst = dsts[i]

pt2 = xyz[i]

print("{} -> {} with d = {}".format(src, tgt, dst))


# In[44]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pt0 = grcs[src]
pt1 = gcts[src]
ax.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], [pt0[2], pt1[2]], 'o-k')
ax.plot([pt1[0], pt1[0]+1e3], [pt1[1], pt1[1]], [pt1[2], pt1[2]], '-k')
ax.plot([pt1[0], pt1[0]-1e3], [pt1[1], pt1[1]], [pt1[2], pt1[2]], '-k')
ax.plot(adend[tgt][:,0], adend[tgt][:,1], adend[tgt][:,2],'.r')
ax.plot(bdend[tgt][:,0], bdend[tgt][:,1], bdend[tgt][:,2],'.m')
pt2 = xyz[i]
ax.plot([pt2[0]], [pt2[1]], [pt2[2]], 'oc')
# ax.set_xlim([0, 1500])


# In[46]:


get_ipython().run_line_magic('matplotlib', 'nbagg')

fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=False)
ax[0].plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 'o-k')
ax[0].plot([pt1[0], pt1[0]+1e3], [pt1[1], pt1[1]], '-k')
ax[0].plot([pt1[0], pt1[0]-1e3], [pt1[1], pt1[1]], '-k')
ax[0].plot(adend[tgt][:,0], adend[tgt][:,1], '.r')
ax[0].plot(bdend[tgt][:,0], bdend[tgt][:,1], '.m')
ax[0].plot([pt2[0]], [pt2[1]], 'oc')


ax[1].plot([pt0[0], pt1[0]], [pt0[2], pt1[2]], 'o-k')
ax[1].plot([pt1[0], pt1[0]+1e3], [pt1[2], pt1[2]], '-k')
ax[1].plot([pt1[0], pt1[0]-1e3], [pt1[2], pt1[2]], '-k')
ax[1].plot(adend[tgt][:,0], adend[tgt][:,2],'.r')
ax[1].plot(bdend[tgt][:,0], bdend[tgt][:,2],'.m')
ax[1].plot([pt2[0]], [pt2[2]], 'oc')


# ## Ascending axon - GoC

# In[50]:


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


# In[56]:


i = 20
tgt = tgts[i]
src = srcs[i]
dst = dsts[i]

pt2 = xyz[i]

print("{} -> {} with d = {}".format(src, tgt, dst))


# In[57]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pt0 = grcs[src]
pt1 = gcts[src]
ax.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], [pt0[2], pt1[2]], 'o-k')
ax.plot([pt1[0], pt1[0]+1e3], [pt1[1], pt1[1]], [pt1[2], pt1[2]], '-k')
ax.plot([pt1[0], pt1[0]-1e3], [pt1[1], pt1[1]], [pt1[2], pt1[2]], '-k')
ax.plot(adend[tgt][:,0], adend[tgt][:,1], adend[tgt][:,2],'.r')
ax.plot(bdend[tgt][:,0], bdend[tgt][:,1], bdend[tgt][:,2],'.m')
pt2 = xyz[i]
ax.plot([pt2[0]], [pt2[1]], [pt2[2]], 'oc')
ax.set_xlim([0, 1500])


# In[59]:


get_ipython().run_line_magic('matplotlib', 'nbagg')

fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=True)
ax[0].plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 'o-k')
ax[0].plot([pt1[0], pt1[0]+1e3], [pt1[1], pt1[1]], '-k')
ax[0].plot([pt1[0], pt1[0]-1e3], [pt1[1], pt1[1]], '-k')
ax[0].plot(adend[tgt][:,0], adend[tgt][:,1], '.r')
ax[0].plot(bdend[tgt][:,0], bdend[tgt][:,1], '.m')
ax[0].plot([pt2[0]], [pt2[1]], 'oc')


ax[1].plot([pt0[0], pt1[0]], [pt0[2], pt1[2]], 'o-k')
ax[1].plot([pt1[0], pt1[0]+1e3], [pt1[2], pt1[2]], '-k')
ax[1].plot([pt1[0], pt1[0]-1e3], [pt1[2], pt1[2]], '-k')
ax[1].plot(adend[tgt][:,0], adend[tgt][:,2],'.r')
ax[1].plot(bdend[tgt][:,0], bdend[tgt][:,2],'.m')
ax[1].plot([pt2[0]], [pt2[2]], 'oc')


# ## Convert big files to the npy format

# In[64]:


src = np.loadtxt(output_path.parent / "PFtoGoCsources.dat")
tgt = np.loadtxt(output_path.parent / "PFtoGoCtargets.dat")
src = src.astype(int)
tgt = tgt.astype(int)


# In[82]:


np.save(output_path.parent / 'PFtoGoCsources.npy', src)
np.save(output_path.parent / 'PFtoGoCtargets.npy', tgt)


# In[85]:


src = np.loadtxt(output_path.parent / "AAtoGoCsources.dat")
tgt = np.loadtxt(output_path.parent / "AAtoGoCtargets.dat")
src = src.astype(int)
tgt = tgt.astype(int)

np.save(output_path.parent / 'AAtoGoCsources.npy', src)
np.save(output_path.parent / 'AAtoGoCtargets.npy', tgt)


# ## Load back data

# In[ ]:


srcs = np.load(output_path.parent / 'PFtoGoCsources.npy')
tgts = np.load(output_path.parent / 'PFtoGoCtargets.npy')


# In[ ]:


grcs = np.unique(srcs)
gocs = np.unique(tgts)


# In[ ]:


syns_per_goc = [sum(tgts==i) for i in gocs]
    

