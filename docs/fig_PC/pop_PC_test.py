# %%
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append('../../')

# %%
import numpy as np
from neuron import h
from pybrep.pop_generation.ebeida import ebeida_sampling
from pybrep.pop_generation.utils import PointCloud
import matplotlib.pyplot as plt

h.load_file("../../test/set3005/Parameters.hoc")
h.MFxrange = 500
h.MFxrange += 50
h.MFyrange = 500+50
h.GLdepth += 50

# %%
fname = "coords_20191016_PC.npz"

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

spacing_mf = 22
mf_points = ebeida_sampling(mf_box, spacing_mf, n_mf*20, True)
# mf_points = mf_points + np.random.randn(mf_points.shape[0], mf_points.shape[1])*0

np.savez(
    fname,
    mf=mf_points
)
