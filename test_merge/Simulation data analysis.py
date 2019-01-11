
# coding: utf-8

# In[1]:


import datashader as ds

import datashader.transfer_functions as tf
from datashader.bokeh_ext import InteractiveImage

from matplotlib.cm import viridis, jet, magma
from datashader.utils import export_image

from bokeh.plotting import figure, output_notebook, show
output_notebook()


# In[2]:


import pandas as pd
import numpy as np

import data_io as io
from importlib import reload
reload(io)


# In[6]:


out = io.OutputReader('/Users/shhong/Desktop/output.5030124')
grc = out.read_spike_data('grc')
goc = out.read_spike_data('goc')

# goc = io.read_neuron_vectors('/Users/shhong/Documents/Sync/To Mint/output.12097191/GoCspiketime.bin')
# goc_coord = io.read_sorted_coords('/Users/shhong/Documents/Sync/To Mint/output.12097191/GoCcoordinates.sorted.dat')
# goc = io.attach_coords(goc, goc_coord)

# mf = io.read_neuron_vectors('/Users/shhong/Documents/Sync/To Mint/output.12097191/MFspiketime.bin')
# mf_coord = io.read_out_MFcoordinates('/Users/shhong/Documents/Sync/To Mint/output.12097191/MFcoordinates.dat')
# mf = io.attach_coords(mf, mf_coord)


# In[7]:


grc3 = grc
#grc3 = grc[(grc.x>100)*(grc.x<1400)*(grc.y>0)*(grc.y<600)]

y_range = (0, 700)
x_range = (300, 800)

def base_plot(tools='pan,box_zoom,wheel_zoom,reset,save'):
    p = figure(tools=tools, plot_width=900, plot_height=400, x_range=x_range, y_range=y_range)
    p.axis.visible = True
    p.xaxis.axis_label = 'Time (ms)'
    p.yaxis.axis_label = 'y (um)'
    return p

def image_callback(x_range, y_range, w, h):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(grc3, 'time', 'y')
    img = tf.shade(agg, cmap=magma, how='eq_hist')
    return tf.dynspread(img)

output_notebook()
p = base_plot()
InteractiveImage(p, image_callback)


# In[8]:


grc3 = grc
#grc3 = grc[(grc.x>100)*(grc.x<1400)*(grc.y>0)*(grc.y<600)]

y_range = (0, 1500)
x_range = (300, 800)

def base_plot(tools='pan,box_zoom,wheel_zoom,reset,save'):
    p = figure(tools=tools, plot_width=900, plot_height=400, x_range=x_range, y_range=y_range)
    p.axis.visible = True
    p.xaxis.axis_label = 'Time (ms)'
    p.yaxis.axis_label = 'x (um)'
    return p

def create_image(x_range, y_range, w, h):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(grc3, 'time', 'x')
    img = tf.shade(agg, cmap=magma, how='eq_hist')
    return tf.dynspread(img)

p = base_plot()
InteractiveImage(p, create_image)


# In[9]:


cc = grc
df2 = cc[np.logical_and(cc['time']>500, cc['time']<4000)]
y_range = (0, 700)
x_range = (0, 1500)

def base_plot(tools='pan,box_zoom,xwheel_zoom,reset,save'):
    p = figure(tools=tools, plot_width=int(700*1.25), plot_height=int(350*1.25), x_range=x_range, y_range=y_range)
    p.axis.visible = True
    p.xaxis.axis_label = 'x (um)'
    p.yaxis.axis_label = 'y (um)'
    return p

def create_image(data, x_range, y_range, w, h):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(data, 'x', 'y')
    img = tf.shade(agg, cmap=magma, how='eq_hist')
#     agg = cvs.points(data, 'x', 'y', ds.count_cat('celltype'))
#     img = tf.shade(agg, color_key={'goc':'red', 'grc':'blue'}, how='log')
    return tf.dynspread(img)

from functools import partial
create_image2 = partial(create_image, df2)

p = base_plot()
InteractiveImage(p, create_image2)


# In[13]:


grc3 = goc
#grc3 = grc[(grc.x>100)*(grc.x<1400)*(grc.y>0)*(grc.y<600)]

y_range = (0, 1500)
x_range = (300, 800)

def base_plot(tools='pan,box_zoom,wheel_zoom,reset,save'):
    p = figure(tools=tools, plot_width=900, plot_height=400, x_range=x_range, y_range=y_range)
    p.axis.visible = True
    p.xaxis.axis_label = 'Time (ms)'
    p.yaxis.axis_label = 'x (um)'
    return p

def create_image(x_range, y_range, w, h):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(grc3, 'time', 'x')
    img = tf.shade(agg, cmap=magma, how='eq_hist')
    return tf.spread(img, px=2)

p = base_plot()
InteractiveImage(p, create_image)


# In[11]:


grc3 = goc
#grc3 = grc[(grc.x>100)*(grc.x<1400)*(grc.y>0)*(grc.y<600)]

y_range = (0, 700)
x_range = (300, 800)

def base_plot(tools='pan,box_zoom,wheel_zoom,reset,save'):
    p = figure(tools=tools, plot_width=900, plot_height=400, x_range=x_range, y_range=y_range)
    p.axis.visible = True
    p.xaxis.axis_label = 'Time (ms)'
    p.yaxis.axis_label = 'y (um)'
    return p

def create_image(x_range, y_range, w, h):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(grc3, 'time', 'y')
    img = tf.shade(agg, cmap=magma, how='eq_hist')
    return tf.spread(img, px=2)

p = base_plot()
InteractiveImage(p, create_image)


# In[12]:


cc = goc
df2 = cc[np.logical_and(cc['time']>500, cc['time']<4000)]
y_range = (0, 700)
x_range = (0, 1500)

def base_plot(tools='pan,box_zoom,xwheel_zoom,reset,save'):
    p = figure(tools=tools, plot_width=int(700*1.25), plot_height=int(350*1.25), x_range=x_range, y_range=y_range)
    p.axis.visible = True
    p.xaxis.axis_label = 'x (um)'
    p.yaxis.axis_label = 'y (um)'
    return p

def create_image(data, x_range, y_range, w, h):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(data, 'x', 'y')
    img = tf.shade(agg, cmap=magma, how='eq_hist')
#     agg = cvs.points(data, 'x', 'y', ds.count_cat('celltype'))
#     img = tf.shade(agg, color_key={'goc':'red', 'grc':'blue'}, how='log')
    return tf.spread(img,px=2)

from functools import partial
create_image2 = partial(create_image, df2)

p = base_plot()
InteractiveImage(p, create_image2)


# In[51]:


def create_image(data, x_range, y_range, w, h):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(data, 'x', 'y', ds.count_cat('celltype'))
    img = tf.shade(agg, color_key={'goc':'red', 'grc': 'blue'}, how='linear')
    return tf.spread(img, px=1)


from tqdm import tqdm
for i in tqdm(range(2000)):
    df2 = df[np.logical_and(df['time']>500+i, df['time']<505+i)]
    img = create_image(df2, x_range, y_range, 700, 350)
#    img = tf.dynspread(img)
    img = tf.spread(img, px=1) 
    export_image(img, 'imgs/%d' % i)


# In[122]:


def create_image(data, x_range, y_range, w, h):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(data, 'time', 'x')
    img = tf.shade(agg, cmap=viridis, how='eq_hist')
    return tf.dynspread(img)


# In[15]:


get_ipython().run_line_magic('pinfo2', 'tf.set_background')


# In[41]:


grc['celltype'] = 'grc'
goc['celltype'] = 'goc'


# In[118]:


df = pd.concat([grc, goc, goc, goc], ignore_index=True)
df.celltype = df.celltype.astype('category')


# In[50]:


xmin = 0
xmax = 1500
ymin = 0
ymax = 700
cc = grc
coords = out.read_sorted_coords('GCcoordinates.sorted.dat')
grc3 = cc[(cc.x>xmin)&(cc.x<xmax)&(cc.y>ymin)&(cc.y<ymax)&(cc.time>1000)]
ncell = coords[(coords.x>xmin)&(coords.x<xmax)&(coords.y>ymin)&(coords.y<ymax)].shape[0]
grc3.shape[0]/2.5/ncell


# In[81]:


coords = out.read_sorted_coords('GCcoordinates.sorted.dat')


# In[93]:


img1 = create_image(goc, (500, 3000), (0, 1500), 750, 350)
img2 = create_image(grc, (500, 3000), (0, 1500), 750, 350)


# In[64]:


import PIL.Image as Image


# In[95]:


Image.blend(img1.to_pil(), img2.to_pil(), 0.5).save('test.png')


# In[83]:


img2.to_pil().save('test.png')


# In[96]:





# In[88]:


get_ipython().run_line_magic('pinfo2', 'output_notebook')

