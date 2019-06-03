# %%
%load_ext autoreload
%autoreload 2
import numpy as np
from neuron import h
import pds_plots as ppl
import matplotlib.pyplot as plt
from pop_generation.mf_generation import Bridson_sampling_2d

h.load_file('test/set3005/Parameters.hoc')

# %%
h.MFxrange = 740
h.MFyrange = 740
h.GLdepth = 240


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
    print('N MF = {}'.format(n_mf))
    return ((Horizontal_range, Transverse_range), n_mf)

mf_box, n_mf = compute_mf_params(h)

spacing_mf = 14.2
mf_points = Bridson_sampling_2d(mf_box, spacing_mf, n_mf, True)

# %%
ppl.plot_mf_1(mf_points, mf_box, 5)
ppl.plot_mf_2(mf_points, mf_box)

# %%
def compute_goc_params(h):
    Transverse_range = h.MFyrange
    Horizontal_range = h.MFxrange
    Vertical_range = h.GoCzrange + 50

    Volume = Transverse_range * Horizontal_range * Vertical_range

    d_goc = h.GoCdensity
    n_goc = int(d_goc * Volume * 1e-9)
    return ((Horizontal_range, Transverse_range, Vertical_range), n_goc)


goc_box, n_goc = compute_goc_params(h)

# %%
# spacing_glo = 6.6-1 #(Billings et al., 2014) Since glomeruli is elipsoid shape, I recalculated based on the spatial occupancy of glomeruli and its density. Also, I subtract 1 cuz I will give Gaussian noise
# spacing_grc = 6-1 # (Billings et al., 2014) I subtract 1 because I will give Gaussian noise

# #Density of cells (mm^-3)
# d_goc = 9500  # (Dugue GP et al., 2009)
# d_glo = 6.6 * 1e5 #(Billings et al., 2014)
# d_grc = 1.9 * 1e6 #(Billings et al., 2014)

spacing_goc = 42 #40 #(NH Barmack, V Yakhnitsa, 2008)

from pop_generation.mf_generation import Bridson_sampling_first
goc_points = Bridson_sampling_first(goc_box, spacing_goc, n_goc, True)
goc_points = goc_points + np.random.normal(0, 1, size=(len(goc_points), 3)) #Gaussian noise
goc_points = goc_points-np.array([20, 20, 20])

# %%
ppl.plot_goc(goc_points, goc_box, 0, 20)

# %%
_ = plt.hist(goc_points[:, 2], 200)

# %%

# %%
print(n_goc, ' ', goc_points.shape[0])

# %%
def compute_glo_params(h):
    Transverse_range = h.MFyrange
    Horizontal_range = h.MFxrange
    Vertical_range = h.GLdepth
    Volume = Transverse_range * Horizontal_range * Vertical_range

    MFdensity = h.MFdensity

    box_fac = 2.5
    Xinstantiate = 64 + 40  # 297+40
    Yinstantiate = 84 + 40 * box_fac  # 474+40x*box_fac

    d_glo = 6.6 * 1e5 #(Billings et al., 2014)
    n_glo = int(d_glo * Volume * 1e-9)
    print("N of Glomeruli = {}".format(n_glo))

    return ((Horizontal_range, Transverse_range//3, Vertical_range), n_glo)

#Glomerulus (Rosettes)
from pop_generation.second_generation import Bridson_sampling_second

globox, n_glo = compute_glo_params(h)
spacing_glo = 6.6-1 #(Billings et al., 2014) Since glomeruli is elipsoid shape, I recalculated based on the spatial occupancy of glomeruli and its density. Also, I subtract 1 cuz I will give Gaussian noise

glo_points = Bridson_sampling_second(globox, spacing_glo, n_glo, True, goc_points)
glo_points[:, 1] = glo_points[:, 1]*3 # Since glomerulus is stretched for Horizontal section, we will generate coordinates in small area at first, and then multiply it with 3. (Billings et al., 2014)

glo_points = glo_points+np.random.normal(0, 0.5, size=(len(glo_points), 3))

# %%
glo_points[:,0].max()
glo_points[:,1].max()
glo_points[:,2].max()

# %%
ppl.plot_glo(glo_points, goc_box, np.array([0, 10])+100, 2.7)

# %%
ppl.plot_goc_glo((goc_points, 15), (glo_points, 2.7), goc_box, np.array([0, 5])+50)

# %%
from pop_generation.third_generation import Bridson_sampling_third

def compute_grc_params(h):
    Transverse_range = h.MFyrange
    Horizontal_range = h.MFxrange
    Vertical_range = h.GLdepth
    Volume = Transverse_range * Horizontal_range * Vertical_range

    MFdensity = h.MFdensity

    box_fac = 2.5
    Xinstantiate = 64 + 40  # 297+40
    Yinstantiate = 84 + 40 * box_fac  # 474+40x*box_fac

    d_grc = 1.9 * 1e6 #(Billings et al., 2014)
    n_grc = int(d_grc * Volume * 1e-9)

    print("N of GrC = {}".format(n_grc))

    return ((Horizontal_range, Transverse_range, Vertical_range), n_grc)


spacing_grc = 6-1 # (Billings et al., 2014) I subtract 1 because I will give Gaussian noise

grcbox, n_grc = compute_grc_params(h)

grc_points = Bridson_sampling_third(grcbox, spacing_grc, n_grc, True, goc_points, glo_points)
ppl.plot_goc_glo((goc_points, 15), (glo_points, 2.7), (glo_points, 2.5), goc_box, np.array([0, 5])+50)
