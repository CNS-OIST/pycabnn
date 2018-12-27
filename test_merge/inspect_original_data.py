
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from joblib import Parallel, delayed


# In[94]:


from pathlib import Path
dataroot = Path('/Users/shhong/Dropbox/network_data/output_brep_2')


# In[15]:


from tqdm import tnrange
aasrc = np.hstack([np.loadtxt(dataroot / 'AAtoGoCsources{}.dat'.format(i)) for i in tnrange(120)])
aatgt = np.hstack([np.loadtxt(dataroot / 'AAtoGoCtargets{}.dat'.format(i)) for i in tnrange(120)])


# In[30]:


np.min(aasrc), np.max(aasrc)


# In[31]:


pfsrc = np.hstack([np.loadtxt(dataroot / 'PFtoGoCsources{}.dat'.format(i)) for i in tnrange(120)])
pftgt = np.hstack([np.loadtxt(dataroot / 'PFtoGoCtargets{}.dat'.format(i)) for i in tnrange(120)])


# In[32]:


np.min(pfsrc), np.max(pfsrc)


# In[33]:


np.min(pftgt), np.max(pftgt)


# In[39]:


naacons = aasrc.size
npfcons = pfsrc.size
naacons, npfcons


# In[43]:


(naacons/(np.max(pftgt)+1), npfcons/(np.max(pftgt)+1))


# In[44]:


(naacons/(np.max(pfsrc)+1), npfcons/(np.max(pfsrc)+1))


# ## BREP 1

# In[45]:


dataroot = Path('/Users/shhong/Dropbox/network_data/output_brep_1')
aasrc = np.hstack([np.loadtxt(dataroot / 'AAtoGoCsources{}.dat'.format(i)) for i in tnrange(120)])
aatgt = np.hstack([np.loadtxt(dataroot / 'AAtoGoCtargets{}.dat'.format(i)) for i in tnrange(120)])
pfsrc = np.hstack([np.loadtxt(dataroot / 'PFtoGoCsources{}.dat'.format(i)) for i in tnrange(120)])
pftgt = np.hstack([np.loadtxt(dataroot / 'PFtoGoCtargets{}.dat'.format(i)) for i in tnrange(120)])


# In[46]:


print((np.min(pfsrc), np.max(pfsrc)))
print((np.min(pftgt), np.max(pftgt)))
naacons = aasrc.size
npfcons = pfsrc.size
naacons, npfcons


# In[49]:


aaseg = np.vstack([np.loadtxt(dataroot / 'AAtoGoCsegments{}.dat'.format(i)) for i in tnrange(120)])


# In[71]:


np.min(aaseg[:,0]), np.max(aaseg[:,0])


# In[72]:


np.min(aaseg[:,1]), np.max(aaseg[:,1])


# In[73]:


plt.hist(aaseg[:,0],np.arange(0,8))


# In[74]:


plt.hist(aaseg[:,1],np.arange(0,7))


# In[76]:


aadst = np.hstack([np.loadtxt(dataroot / 'AAtoGoCdistances{}.dat'.format(i)) for i in tnrange(120)])


# In[78]:


plt.hist(aadst,50)


# In[95]:


pfseg = np.vstack([np.loadtxt(dataroot / 'PFtoGoCsegments{}.dat'.format(i)) for i in tnrange(120)])


# In[96]:


plt.hist(pfseg[:,0],np.arange(0,8))


# In[97]:


plt.hist(pfseg[:,1],np.arange(0,7))


# In[98]:


pfdst = np.hstack([np.loadtxt(dataroot / 'PFtoGoCdistances{}.dat'.format(i)) for i in tnrange(120)])


# In[99]:


plt.hist(pfdst,50)


# ## PyBREP

# In[79]:


dataroot = Path('/Users/shhong/Dropbox/network_data/model_pybrep_output')
aasrc = np.hstack([np.loadtxt(dataroot / 'AAtoGoCsources{}.dat'.format(i)) for i in tnrange(120)])
aatgt = np.hstack([np.loadtxt(dataroot / 'AAtoGoCtargets{}.dat'.format(i)) for i in tnrange(120)])
pfsrc = np.hstack([np.loadtxt(dataroot / 'PFtoGoCsources{}.dat'.format(i)) for i in tnrange(120)])
pftgt = np.hstack([np.loadtxt(dataroot / 'PFtoGoCtargets{}.dat'.format(i)) for i in tnrange(120)])


# In[80]:


np.min(pfsrc), np.max(pfsrc)


# In[81]:


np.min(pftgt), np.max(pftgt)


# In[83]:


naacons = aasrc.size
npfcons = pfsrc.size
naacons, npfcons


# In[84]:


(naacons/(np.max(pftgt)+1), npfcons/(np.max(pftgt)+1))


# In[85]:


(naacons/(np.max(pfsrc)+1), npfcons/(np.max(pfsrc)+1))


# In[86]:


aaseg = np.vstack([np.loadtxt(dataroot / 'AAtoGoCsegments{}.dat'.format(i)) for i in tnrange(120)])


# In[87]:


np.min(aaseg[:,0]), np.max(aaseg[:,0])


# In[88]:


np.min(aaseg[:,1]), np.max(aaseg[:,1])


# In[89]:


plt.hist(aaseg[:,0],np.arange(0,8))


# In[90]:


plt.hist(aaseg[:,1],np.arange(0,7))


# In[92]:


aadst = np.hstack([np.loadtxt(dataroot / 'AAtoGoCdistances{}.dat'.format(i)) for i in tnrange(120)])


# In[93]:


plt.hist(aadst,50)


# In[101]:


pfseg = np.vstack([np.loadtxt(dataroot / 'PFtoGoCsegments{}.dat'.format(i)) for i in tnrange(120)])


# In[102]:


plt.hist(pfseg[:,0],np.arange(0,8))


# In[103]:


plt.hist(pfseg[:,1],np.arange(0,7))


# In[104]:


pfdst = np.hstack([np.loadtxt(dataroot / 'PFtoGoCdistances{}.dat'.format(i)) for i in tnrange(120)])


# In[105]:


plt.hist(pfdst,50)


# In[106]:


min(pfdst)

