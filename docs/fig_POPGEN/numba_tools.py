import numpy as np
import numba as nb
from numba import njit, prange

 @njit(parallel=True)
 def tile2d(x, m, n):
     z = np.empty((x.shape[0]*m, x.shape[1]*n), x.dtype)
     for i in prange(n):
         for j in prange(m):
             z[(j*x.shape[0]):((j+1)*x.shape[0]), (i*x.shape[1]):((i+1)*x.shape)] = x
     return z

# @njit(parallel=True)
# def all_axis(a, axis):
#     if axis==-1:
#         n = a.ndim
#     else:
#         n = axis

#     for i in prange(a)
