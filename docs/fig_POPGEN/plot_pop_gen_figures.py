# %%
%load_ext autoreload
%autoreload 2
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

import pds_plots as ppl

# %%
f = np.load('coords_20190613_1.npz')

mf = f['mf']

bbox = [[0, 1500], [0-20, 700], [0, 200]]

def limit_to_box(x, box):
    mf = x.copy()
    for i, t in enumerate(box):
        mf = mf[mf[:,i]>=t[0], :]
        mf = mf[mf[:,i]<=t[1], :]
    return mf

def fix_coors(x):
    y = x-np.ones(x.shape[1])*25
    box = [bbox[0], [bbox[1][0]+20, bbox[1][1]], bbox[2]]
    return limit_to_box(y, box[:x.shape[1]])

mf = fix_coors(mf)


# %%
bbox = bbox[:2]
ax = ppl.plot_mf_1(mf, bbox, 4, save=False)
ax.plot([bbox[0][1]-100, bbox[0][1]], [bbox[1][0]+2.5, bbox[1][0]+2.5], 'w', linewidth=10)
# ppl.plot_mf_2(mf, [1500, 700], save=True)
plt.tight_layout()
plt.savefig('mf.png', dpi=300/2.54)


# %%
bbox = [[0, 1500], [0-20, 700], [0, 200]]

goc = f['goc']
goc = fix_coors(goc)

ax = ppl.plot_goc(goc, bbox, 100, 13.5)
ax.plot([bbox[0][1]-100, bbox[0][1]], [bbox[1][0], bbox[1][0]], 'w', linewidth=15)
plt.savefig('goc.png', dpi=300/2.54)

# %%
glo = f['glo']
glo = fix_coors(glo)

ax = ppl.plot_goc_glo((goc, 13.5), (glo, 6.6 / 1.75), bbox, 100)
ax.plot([bbox[0][1]-100, bbox[0][1]], [bbox[1][0], bbox[1][0]], 'w', linewidth=15)
plt.savefig('goc+glo.png', dpi=300/2.54)
#         plt.savefig('goc+glo.png', dpi=300

# %%
grc = f['grc_nop']
grc = fix_coors(grc)

ax = ppl.plot_all_pop(
    (goc, 13.5),
    (glo, 6.6 / 1.75),
    (grc, 2.75),
    bbox, 100)
ax.plot([bbox[0][1]-100, bbox[0][1]], [bbox[1][0], bbox[1][0]], 'w', linewidth=15)
plt.savefig('all.png', dpi=300/2.54)


# %%
def to_yzx(x):
    y = x.copy()
    y[:,0] = x[:,1]
    y[:,1] = x[:,2]
    y[:,2] = x[:,0]
    return y

goc = to_yzx(goc)
glo = to_yzx(glo)
grc = to_yzx(grc)

bbox = [[200,400], [0-20, 200]]
ax = ppl.plot_goc(goc, bbox, 700, 17)
ax.plot([bbox[0][1]-100, bbox[0][1]], [bbox[1][0], bbox[1][0]], 'w', linewidth=15)

# %%
ax = ppl.plot_goc_glo((goc, 17), (glo, 6.6 / 1.75), bbox, 700)
ax.plot([bbox[0][1]-100, bbox[0][1]], [bbox[1][0], bbox[1][0]], 'w', linewidth=15)

# %%
ax = ppl.plot_all_pop(
    (goc, 17),
    (glo, 6.6/1.55),
    (grc, 3),
    bbox, 700)
ax.plot([bbox[0][1]-100, bbox[0][1]], [bbox[1][0], bbox[1][0]], 'w', linewidth=15)


# %%

# %%

# %%
