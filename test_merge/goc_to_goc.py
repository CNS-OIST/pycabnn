
# coding: utf-8

# In[241]:


import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# In[2]:


root = Path('/Users/shhong/Dropbox/network_data/output_brep_2')


# In[3]:


root


# In[8]:


xyz = np.loadtxt(root / "GoCcoordinates.sorted.dat")
dist0 = np.loadtxt(root / "GoCtoGoCdistances.dat")


# In[38]:


src = np.loadtxt(root / "GoCtoGoCsources.dat").astype(int)
tgt = np.loadtxt(root / "GoCtoGoCtargets.dat").astype(int)


# In[11]:


axon0 = np.loadtxt(root / "GoCaxoncoordinates.sorted.dat")


# In[26]:


axon0.shape
npoints = int(axon0.shape[1]/3)


# In[270]:


# axon = [np.vstack([x[:3], x.reshape(npoints, 3)[1::2]]) for x in axon0]
axon = [x.reshape(npoints, 3)[1::2] for x in axon0]
axon[0]


# In[271]:


from tqdm import tqdm_notebook
colx = {}
for i in tqdm_notebook(range(ncell)):
    for j in range(ncell):
        if i!=j:
#         i = 1759
#         j = 1936
            di, ii = cKDTree(axon[i]).query(xyz[j])
            axon_len = np.sqrt(np.sum((axon[i][ii]-xyz[i])**2))
#             print(di, axon_len, axon_len+di)
            if di<100:
                colx[(i, j)] = axon_len + di
            


# In[150]:


il, = np.where(np.logical_and(src==i, tgt==j))
il


# In[151]:


dist0[il]


# In[88]:


axon[[1,2]]


# In[96]:


ncell = 1995
ncoords = int(axon.shape[0]/ncell)


# In[111]:


cellind!=0


# In[108]:


cellind.shape


# In[109]:


axon.shape


# In[113]:


xyz[0]


# In[119]:


cellind[cellind!=i][1173]


# In[159]:


len(colx.keys())


# In[160]:


src.size


# In[256]:


i = 0
s, t = src[i], tgt[i]
print(s, t)
print(colx[(s, t)])
print(dist0[i])


# In[225]:


colx[(t, s)]


# In[261]:


s, t
di, ii = cKDTree(axon[s]).query(xyz[t])
axon_len = np.sqrt(np.sum((axon[s][ii]-xyz[s])**2))
print(s, t, di, axon_len)


# In[235]:


dd[np.logical_and(dd[:,0]==s, dd[:,1]==t),2]


# In[275]:


dd = np.array([colx[(src[i], tgt[i])]-dist0[i] for i, _ in enumerate(src) if (src[i], tgt[i]) in colx.keys()])


# In[276]:


_ = plt.hist(dd,100)


# In[277]:


plt.plot(np.sort(dd),'.')


# # ??

# In[182]:


xyz[t]


# In[189]:


dd = np.loadtxt(root / "GoCdistances0.dat")


# In[238]:


plt.plot(dd)


# In[273]:


len(colx.keys())


# In[278]:


xyz = np.loadtxt(root / "GoCcoordinates.sorted.dat")


# In[279]:


from neuron import h


# In[280]:


h.load_file('/Users/shhong/Documents/cerebellar_cortex/Molecular_Layer/params/set3005/Parameters.hoc')


# In[285]:


width = []
minp = []
for k in ['X', 'Y', 'Z']:
    amin = eval("h.GoC_Axon_{}min".format(k))
    amax = eval("h.GoC_Axon_{}max".format(k))
    width.append(amax-amin)
    minp.append(amin)
width, minp = np.array(width), np.array(minp)


# In[327]:


axon_coord = np.random.rand(ncell*20,3)
axon_coord = axon_coord*width + minp


# In[328]:


naxon = int(h.numAxonGolgi)
axon_base = np.zeros_like(axon_coord)
for i in range(ncell):
    axon_coord[(naxon*i):(naxon*(i+1))] = axon_coord[(naxon*i):(naxon*(i+1))] + xyz[i]
    axon_base[(naxon*i):(naxon*(i+1))] = xyz[i]
axon_coord


# In[329]:


axon_coord_file = np.hstack([axon_base, axon_coord])
axon_coord_file


# In[337]:


temp = np.reshape(axon_coord_file, (ncell,  2*naxon*3))
temp[0,:]
np.savetxt('temp.dat', temp)


# In[338]:


axon_coord


# In[363]:


from tqdm import tqdm_notebook
dist = []
src = []
tgt = []

for i in tqdm_notebook(range(ncell)):
    axon_coord1 = axon_coord[(i*naxon):((i+1)*naxon),:]
    tree = cKDTree(axon_coord1)
    for j in range(ncell):
        if i!=j:
            di, ii=tree.query(xyz[j])
            axon_len = np.linalg.norm(axon_coord1[ii]-xyz[i])
            if di<h.GoCtoGoCzone:
                src.append(i)
                tgt.append(j)
                dist.append(axon_len + di)
            


# In[364]:


dist


# In[340]:


axon_coord[0:2*naxon]


# In[353]:


np.savetxt('temp.dat', tgt, fmt='%d')


# In[369]:


np.savetxt('temp.dat', dist)


# In[356]:


from tqdm import tqdm_notebook
dist = []
src = []
tgt = []
for i in tqdm_notebook(range(ncell)):
    for j in range(ncell):
        if i!=j:
            di = np.linalg.norm(xyz[j]-xyz[i])
            if di<h.GoCtoGoCgapzone:
                src.append(i)
                tgt.append(j)
                dist.append(di)


# In[357]:


dist


# In[370]:


z = h.File('temp.dat')
z.ropen()


# In[371]:


v = h.Vector()
v.scanf(z)


# In[372]:


v.x[0]


# In[373]:


z.close()


# In[376]:


v.x[1]


# In[377]:


get_ipython().system('open .')

