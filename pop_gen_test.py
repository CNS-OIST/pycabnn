# %%
import numpy as np
from neuron import h
import pds_plots as pl
from pop_generation.mf_generation import Bridson_sampling_2d

h.load_file('test_merge/set3005/Parameters.hoc')


# %%
def compute_mf_params(h):
    Transverse_range = h.MFyrange
    Horizontal_range = h.MFxrange
    Vertical_range = h.GLdepth
    Volume = Transverse_range * Horizontal_range * Vertical_range

    MFdensity = h.MFdensity

    box_fac = 2.5
    Xinstantiate = 64 + 40  # 297+40
    Yinstantiate = 84 + 40 * box_fac  # 474+40x*box_fac


    n_mf = int((Transverse_range + (2 * Xinstantiate)) * (Horizontal_range + (2 * Yinstantiate)) * MFdensity * 1e-6)
    return ((Horizontal_range, Transverse_range), n_mf)


mf_box, n_mf = compute_mf_params(h)

# %%
spacing_mf = 14
mf_points = Bridson_sampling_2d(mf_box, spacing_mf, n_mf, True)

# %%
pl.plot_mf_1(mf_points, mf_box)
pl.plot_mf_2(mf_points, mf_box)
