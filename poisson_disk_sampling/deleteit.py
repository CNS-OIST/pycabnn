import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import paired_distances

MIL = np.loadtxt('Interneurons.txt')
MIL = MIL/2.0

min_x = min(MIL[:, 0])
max_x = max(MIL[:, 0])
min_z = min(MIL[:, 1])
max_z = max(MIL[:, 1])
min_y = min(MIL[:, 2])
max_y = max(MIL[:, 2])

plt.Circle((MIL[5, 0], MIL[5, 1]), 1, color='b')
plt.show()
