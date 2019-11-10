#!/usr/bin/env python

import numpy as np
import time

import ipyparallel as ipp
rc = ipp.Client()
dv = rc[:]
lv = rc.load_balanced_view()

r = np.random.randn(5000, 5000)
# rc['r'] = r

# See https://stackoverflow.com/a/17846726
with dv.sync_imports():
    import mymod
fmul1 = lambda b: mymod.fmul(r, b)

Nrepeat = 1000

# Serial version
t = time.process_time()
z = [mymod.fmul(r, i) for i in range(Nrepeat)]
print('Serial version: elapsed time =', time.process_time() - t)

# Parallel version
t = time.process_time()

# See https://stackoverflow.com/a/17846726
dv.block = True
dv.push(dict(r=r))
print('Parallel version (scattering common variables): elapsed time =', time.process_time() - t)
dv.block = False

z = list(lv.map(fmul1, range(Nrepeat)))
print('Parallel version (total): elapsed time =', time.process_time() - t)

# t = time.process_time()
# z = Parallel(n_jobs=10)(delayed(fmul)(r, i) for i in range(Nrepeat))
# print('Joblib parallel version 1: elapsed time =', time.process_time() - t)

# t = time.process_time()
# with Parallel(n_jobs=10) as parallel:
#     r1 = np.random.randn(5000, 5000)
#     z = parallel(delayed(fmul)(r1, i) for i in range(Nrepeat))
# print('Joblib parallel version 2: elapsed time =', time.process_time() - t)

