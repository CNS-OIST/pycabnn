import numpy as np
from GL_poisson import Bridson_sampling
from testpoisson import Bridson_sampling_2
from joblib import Parallel, delayed
import csv
from scipy import spatial
import matplotlib.pyplot as plt
from tqdm import tqdm

Transverse_range = 1500  # Y range
Horizontal_range = 700  # X range
Vertical_range = 200  # 140 for daria

ss = np.arange(5, 7, 0.1)
def test_points(s):
    points = Bridson_sampling((int(Horizontal_range/3), Transverse_range, Vertical_range), s, 138600, True)
    points[:, 0] = points[:, 0]*3
    grc_points = Bridson_sampling_2((Horizontal_range, Transverse_range, Vertical_range), 5, 415800, True, points)
    np.savetxt('s:{} Glomeruli'.format(s), points)
    np.savetxt('s:{} Grc'.format(s), grc_points)

Parallel(n_jobs=-1)(delayed(test_points)(s) for s in ss)