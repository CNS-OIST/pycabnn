
# coding: utf-8

# In[1]:





# In[10]:


import pybrep.cell_population as cell_pop


# In[17]:


import numpy as np


# In[19]:


xyz = np.random.rand(10,3)
xyz[:,0] = xyz[:,0]*1500
xyz[:,1] = xyz[:,1]*750
xyz[:,2] = xyz[:,2]*200


# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'ipympl')


# In[20]:


plt.scatter(xyz[:,0], xyz[:,1], 80)


# In[22]:


mlis = cell_pop.MLI_pop([])
mlis.load_somata(xyz)


# In[24]:


mlis.add_dendrites()


# In[28]:


idx = mlis.qpts.idx
for i in np.unique(idx):
    ii = (i==idx)
    xyz1 = mlis.qpts.coo[ii,:]
    plt.plot(xyz1[:,1], xyz1[:,2], '.')
plt.scatter(xyz[:,1], xyz[:,2], 80)


# In[1]:


cd ..


# In[2]:


import pybrep as brp
import numpy as np
from neuron import h


# In[3]:


h.xopen("/Users/shhong/Dropbox/network_data/model/params/set3005/Parameters.hoc")
gp = brp.create_population('Granule', h)


# In[4]:


gp.load_somata('/Users/shhong/Dropbox/network_data/input_brep_2/GCcoordinates.dat')
gp.add_aa_endpoints_fixed()
gp.add_pf_endpoints()


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

_, ax = plt.subplots(figsize=(20,10))
ax.plot(gp.som[:,0], gp.som[:,1], '.')


# In[9]:


n_mli = 21735 # rat
n_mli = 22275 # human
n_mli = 33413
n_mli = int(33413/10) # test only with 10%


# In[10]:


xyz = np.random.rand(n_mli, 3)
xyz[:,0] = xyz[:,0]*1500
xyz[:,1] = xyz[:,1]*750
xyz[:,2] = xyz[:,2]*200


# In[11]:


_, ax = plt.subplots(figsize=(20,10))
ax.plot(xyz[:,0], xyz[:,1], '.')


# In[12]:


mlip = brp.create_population('MLI', h)


# In[13]:


mlip.load_somata(xyz)


# In[14]:


mlip.add_dendrites()
mlip.qpts.coo[:,2] = mlip.qpts.coo[:,2] + h.GLdepth + h.PCLdepth


# In[22]:


mlip.qpts.idx


# In[30]:


_, ax = plt.subplots(figsize=(20,10))
idx = mlip.qpts.idx
for i in range(300):
    ii = (i==idx)
    xyz1 = mlip.qpts.coo[ii,:]
    ax.plot(xyz1[:,0], xyz1[:,1], '.')
ax.scatter(xyz[:,0], xyz[:,1], 10)


# In[51]:


c_rad_aa = h.AAtoGoCzone/1.73


# In[52]:


cc = brp.Connect_2D(gp.qpts_aa, mlip.qpts, c_rad_aa)


# In[53]:


cc.connections_parallel(deparallelize=True, nblocks=120, debug=True)


# In[54]:


c_rad_pf = h.PFtoGoCzone/1.113
cc = brp.Connect_2D(gp.qpts_pf, mlip.qpts, c_rad_pf)
cc.connections_parallel(deparallelize=True, nblocks=120, debug=True)


# In[48]:


gp.qpts_pf.coo[2][:,2][0]


# In[55]:


cc.save_result('PFtoMLI')


# In[50]:


mlip.qpts.coo


# In[63]:


cc.result.shape[0]/10**6


# In[66]:


tgt = cc.result['target'].values


# In[70]:


z = plt.hist(tgt, range(3341))


# In[73]:


z[0].mean()


# In[74]:


from tqdm import tqdm


# In[76]:


for i, x in tqdm(enumerate(['a', 'b', 'c'])):
    print(i, x)

