import sys
sys.path.append('../../')

import numpy as np
from neuron import h
from pybrep.pop_generation.ebeida import ebeida_sampling
from pybrep.pop_generation.utils import PointCloud
import matplotlib.pyplot as plt

h.load_file("../../test/set3005/Parameters.hoc")
h.MFxrange = 700
h.MFxrange += 50
h.MFyrange += 50
h.GLdepth += 50

f_input_name = "coords_20190626_1.npz"
f_output_name = "coords_20190626_1_1.npz"

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


# %%
def compute_goc_params(h):
    Transverse_range = h.MFyrange
    Horizontal_range = h.MFxrange
    Vertical_range = h.GLdepth + 50

    Volume = Transverse_range * Horizontal_range * Vertical_range

    d_goc = h.GoCdensity
    n_goc = int(d_goc * Volume * 1e-9)
    print("N GoC = {}".format(n_goc))
    return ((Horizontal_range, Transverse_range, Vertical_range), n_goc)


goc_box, n_goc = compute_goc_params(h)
spacing_goc = 45 - 1  # 40 #(NH Barmack, V Yakhnitsa, 2008)
# spacing_goc = 42.5 - 1  # 40 #(NH Barmack, V Yakhnitsa, 2008)


# # %%

scale_factor = 1/3 #0.29/0.75

class GoC(PointCloud):
    def test_points(self, x):
        y = x.copy()
        y[:, 1] = y[:, 1] / scale_factor
        return super().test_points(y)

    def test_cells(self, cell_corners, dgrid):
        y = cell_corners.copy()
        y[:, 1] = y[:, 1] / scale_factor
        return super().test_cells(y, dgrid)

d_goc_glo = 27 / 2 + (7.6) / 2 - 1 + 1/scale_factor

def compute_glo_params(h):
    Transverse_range = h.MFyrange
    Horizontal_range = h.MFxrange
    Vertical_range = h.GLdepth
    Volume = Transverse_range * Horizontal_range * Vertical_range

    MFdensity = h.MFdensity

    box_fac = 2.5
    Xinstantiate = 64 + 40  # 297+40
    Yinstantiate = 84 + 40 * box_fac  # 474+40x*box_fac

    d_grc = 1.9 * 1e6  # (Billings et al., 2014)
    d_glo = d_grc*0.3
#     d_glo = 6.6 * 1e5  # (Billings et al., 2014)
    n_glo = int(d_glo * Volume * 1e-9)
    print("N of Glomeruli = {}".format(n_glo))

    return ((Horizontal_range, int(Transverse_range*scale_factor+0.5) , Vertical_range), n_glo)


# Glomerulus (Rosettes)
globox, n_glo = compute_glo_params(h)

spacing_glo = 8.39 - 1

# %%

f_input = np.load(f_input_name)
mf_points   = f_input['mf']
goc_points  = f_input['goc']
glo_points1 = f_input['glo']

# %%
d_goc_grc = 27 / 2 + 6.7 / 2 + 1
goc = PointCloud(goc_points, d_goc_grc)

d_glo_grc = 7.6 / 2 + 6.7 / 2
glo = PointCloud(glo_points1, d_glo_grc)

def compute_grc_params(h):
    Transverse_range = h.MFyrange
    Horizontal_range = h.MFxrange
    Vertical_range = h.GLdepth
    Volume = Transverse_range * Horizontal_range * Vertical_range

    MFdensity = h.MFdensity

    box_fac = 2.5
    Xinstantiate = 64 + 40  # 297+40
    Yinstantiate = 84 + 40 * box_fac  # 474+40x*box_fac

    d_grc = 1.9 * 1e6  # (Billings et al., 2014)
    n_grc = int(d_grc * Volume * 1e-9)

    print("N of GrC = {}".format(n_grc))

    return ((Horizontal_range, Transverse_range, Vertical_range), n_grc)


# (Billings et al., 2014) I subtract 1 because I will give Gaussian noise
spacing_grc = 6.7 - 1

grcbox, n_grc = compute_grc_params(h)
grc_points = ebeida_sampling(grcbox, spacing_grc, n_grc, True, ftests=[goc, glo])

# %%
np.savez(
    f_output_name,
    mf=mf_points,
    goc=goc_points,
    glo=glo_points1,
    grc_nop=grc_points,
)
