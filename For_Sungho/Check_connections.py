
# coding: utf-8

# ## Parallel fiber - GoC

# In[1]:


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[8]:


output_path = Path.cwd().parent.parent / 'output_1'


# In[9]:


adend = np.loadtxt('/Users/shhong/Documents/Ines/output_1/GoCadendcoordinates.dat')
bdend = np.loadtxt('/Users/shhong/Documents/Ines/output_1/GoCbdendcoordinates.dat')
grcs = np.loadtxt(output_path / "GCcoordinates.sorted.dat")
gcts = np.loadtxt(output_path / "GCTcoordinates.dat")


# In[10]:


adend = adend.reshape((1995,50,3))
bdend = bdend.reshape((1995,24,3))


# In[11]:


get_ipython().run_line_magic('matplotlib', 'nbagg')
i = 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(adend[i][:,0], adend[i][:,1], adend[i][:,2],'.r')
ax.plot(bdend[i][:,0], bdend[i][:,1], bdend[i][:,2],'.k')

i = 1000
ax.plot(adend[i][:,0], adend[i][:,1], adend[i][:,2],'.r')
ax.plot(bdend[i][:,0], bdend[i][:,1], bdend[i][:,2],'.k')


# In[3]:


data_index = 20

data_dir = Path("/Users/shhong/Documents/Ines/output_1")
fcoords = "PFtoGoCcoords{}.dat".format(data_index)
fsrcs = "PFtoGoCsources{}.dat".format(data_index)
ftgts = "PFtoGoCtargets{}.dat".format(data_index)

xyz = np.loadtxt(data_dir / fcoords)
srcs = np.loadtxt(data_dir / fsrcs).astype(int)
tgts = np.loadtxt(data_dir / ftgts).astype(int)


# In[4]:


i = 5000
tgt = tgts[i]
src = srcs[i]
print("{} -> {}".format(src, tgt))


# In[52]:


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


# ## Ascending axon - GoC

# In[12]:


data_index = 80

data_dir = Path("/Users/shhong/Documents/Ines/output_1")
fcoords = "AAtoGoCcoords{}.dat".format(data_index)
fsrcs = "AAtoGoCsources{}.dat".format(data_index)
ftgts = "AAtoGoCtargets{}.dat".format(data_index)

xyz = np.loadtxt(data_dir / fcoords)
srcs = np.loadtxt(data_dir / fsrcs).astype(int)
tgts = np.loadtxt(data_dir / ftgts).astype(int)


# In[13]:


i = 30000
tgt = tgts[i]
src = srcs[i]
print("{} -> {}".format(src, tgt))


# In[18]:


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


# In[15]:


pt2


# In[16]:


tgt

