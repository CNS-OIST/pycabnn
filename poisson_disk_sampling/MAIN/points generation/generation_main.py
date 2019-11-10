from first_generation import Bridson_sampling_first
from second_generation import Bridson_sampling_second
from third_generation import Bridson_sampling_third
from mf_generation import Bridson_sampling_mf
import numpy as np

# Network Architecture Size
Transverse_range = 1500
Horizontal_range = 700
Vertical_range = 200
Volume = Transverse_range * Horizontal_range * Vertical_range

#Mossy Fiber Info
MFdensity = 1650
box_fac = 2.5
Xinstantiate = 64 + 40  # 297+40
Yinstantiate = 84 + 40 * box_fac  # 474+40x*box_fac

#Minimum distance with the same cell (It does not mean a minimum distance with other type of cell)
spacing_mf = 8
spacing_goc = 40 #(NH Barmack, V Yakhnitsa, 2008)
spacing_glo = 6.6-1 #(Billings et al., 2014) Since glomeruli is elipsoid shape, I recalculated based on the spatial occupancy of glomeruli and its density. Also, I subtract 1 cuz I will give Gaussian noise
spacing_grc = 6-1 # (Billings et al., 2014) I subtract 1 because I will give Gaussian noise

#Density of cells (mm^-3)
d_goc = 9500  # (Dugue GP et al., 2009)
d_glo = 6.6 * 1e5 #(Billings et al., 2014)
d_grc = 1.9 * 1e6 #(Billings et al., 2014)

#Number of cells
n_mf = int((Transverse_range + (2 * Xinstantiate)) * (Horizontal_range + (2 * Yinstantiate)) * MFdensity * 1e-6)
n_goc = int(d_goc * Volume * 1e-9)
n_glo = int(d_glo * Volume * 1e-9)
n_grc = int(d_grc * Volume * 1e-9)

##Points Generation

#Mossy fiber
mf_points = Bridson_sampling_mf((Horizontal_range, Transverse_range), spacing_mf, n_mf, True)
mf_points = np.hstack((mf_points, np.reshape(np.random.uniform(50, 150, len(mf_points)), (-1, 1)))) # Randomly make z coordinates
np.savetxt('mf_coordinates.txt', mf_points)

#Golgi Cell
goc_points = Bridson_sampling_first((Horizontal_range, Transverse_range, Vertical_range), spacing_goc, n_goc, True)
goc_points = goc_points + np.random.normal(0, 1, size=(len(goc_points), 3)) #Gaussian noise
np.savetxt('goc_coordinates.txt', goc_points)

#Glomerulus (Rosettes)
glo_points = Bridson_sampling_second((Horizontal_range//3, Transverse_range, Vertical_range), spacing_glo, n_glo, True, goc_points)

    #Since glomerulus is stretched for Horizontal section, we will generate coordinates in small area at first, and then multiply it with 3. (Billings et al., 2014)
glo_points[:, 0] = glo_points[:, 0]*3
glo_points = glo_points+np.random.normal(0, 1, size=(len(glo_points), 3))
np.savetxt('glo_coordinates.txt', glo_points)

#Granular Cell
grc_points = Bridson_sampling_third((Horizontal_range, Transverse_range, Vertical_range), spacing_grc, n_grc, True, goc_points, glo_points)
grc_points = grc_points+np.random.normal(0, 1, size=(len(grc_points), 3))
np.savetxt('grc_coordinates.txt', grc_points)



