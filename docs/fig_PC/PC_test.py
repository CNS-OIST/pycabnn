# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
%load_ext autoreload
%autoreload 2
%matplotlib inline

import sys
sys.path.append('../../')

# %%
import numpy as np
from neuron import h
from pybrep.pop_generation.ebeida import ebeida_sampling
from pybrep.pop_generation.bridson import bridson_sampling
from pybrep.pop_generation.utils import PointCloud
import matplotlib.pyplot as plt

# %%
h.load_file("../../test/set3005/Parameters.hoc")
h.MFxrange = 200
h.MFyrange = 200
h.GLdepth += 50

def compute_mf_params(h):
    Transverse_range = h.MFyrange
    Horizontal_range = h.MFxrange
    Vertical_range = h.GLdepth
    Volume = Transverse_range * Horizontal_range * Vertical_range

    MFdensity = h.MFdensity

    box_fac = 2.5
    Xinstantiate = 64 + 40  # 297+40
    Yinstantiate = 84 + 40 * box_fac  # 474+40x*box_fac

    n_mf = int(
        (Transverse_range + (2 * Xinstantiate))
        * (Horizontal_range + (2 * Yinstantiate))
        * MFdensity
        * 1e-6
    )
    print("N MF = {}".format(n_mf))
    return (
        (Horizontal_range + (2 * Yinstantiate), Transverse_range + (2 * Xinstantiate)),
        n_mf,
    )


mf_box, n_mf = compute_mf_params(h)
mf_box = (200, 200)

# %%
spacing_mf = 20

mf_points = ebeida_sampling(mf_box, spacing_mf, n_mf+500, True)
# mf_points = Bridson_sampling_2d(mf_box, spacing_mf, 82, True)

# mf_points = mf_points + np.random.randn(mf_points.shape[0], mf_points.shape[1])*0

# %%
x = mf_points
_, ax = plt.subplots(ncols=2, figsize=(16,8))

ax[0].scatter(x[:,0], x[:,1], 100)

ax[1].scatter(x[:,0], x[:,1], 1000)
ax[1].scatter(x[:,0], x[:,1], 50)

for i in range(12):
    ax[1].plot([0, h.MFxrange+20], [20*i, 20*i], 'k')
for i in range(12):
    ax[1].plot([20*i, 20*i], [0, h.MFyrange+20], 'k')



# %%
