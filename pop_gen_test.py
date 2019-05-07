# %%
%load_ext autoreload
%autoreload 2
import numpy as np
from neuron import h
import pds_plots as ppl
import matplotlib.pyplot as plt
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
ppl.plot_mf_1(mf_points, mf_box, 5)
ppl.plot_mf_2(mf_points, mf_box)

# %%
def compute_goc_params(h):
    Transverse_range = h.GoCyrange
    Horizontal_range = h.GoCxrange
    Vertical_range = h.GoCzrange

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

spacing_goc = 32 #40 #(NH Barmack, V Yakhnitsa, 2008)

from pop_generation.mf_generation import Bridson_sampling_first
goc_points = Bridson_sampling_first(goc_box, spacing_goc, n_goc, True)
goc_points = goc_points + np.random.normal(0, 1, size=(len(goc_points), 3)) #Gaussian noise

# %%
ppl.plot_goc(goc_points, goc_box, np.array([0, 20])+100, 12)

# %%
plt.hist(goc_points[:, 2], 50)

#%%
print(n_goc, ' ', goc_points.shape[0])

#%%
def compute_glo_params(h):
    Transverse_range = h.MFyrange//3
    Horizontal_range = h.MFxrange
    Vertical_range = h.GLdepth
    Volume = Transverse_range * Horizontal_range * Vertical_range

    MFdensity = h.MFdensity

    box_fac = 2.5
    Xinstantiate = 64 + 40  # 297+40
    Yinstantiate = 84 + 40 * box_fac  # 474+40x*box_fac

    n_mf = int((Transverse_range + (2 * Xinstantiate)) * (Horizontal_range + (2 * Yinstantiate)) * MFdensity * 1e-6)
    return ((Horizontal_range, Transverse_range), n_mf)

#Glomerulus (Rosettes)
glo_points = Bridson_sampling_second((Horizontal_range//3, Transverse_range, Vertical_range), spacing_glo, n_glo, True, goc_points)

    #Since glomerulus is stretched for Horizontal section, we will generate coordinates in small area at first, and then multiply it with 3. (Billings et al., 2014)
glo_points[:, 0] = glo_points[:, 0]*3
glo_points = glo_points+np.random.normal(0, 1, size=(len(glo_points), 3))
np.savetxt('glo_coordinates.txt', glo_points)
