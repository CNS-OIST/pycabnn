import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

r = np.arange(874, 874*2, 100)

def xy_gaussian(x):
    m = 25
    s = 75
    if x == 0:
        return 0
    else:
        return -0.45 + 200 * np.exp((-1 / 2) * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def z_gaussian(x):
    m = 0
    s = 18
    if x == 0:
        return 0
    else:
        return 25 * np.exp((-1 / 2) * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

zero_series = []
one_series = []
two_series = []
three_series = []

def main(i):
    trial = i
    # num = 5
    # while num >0:
    r_x = np.random.uniform(-124.5, 119.75, size=trial)
    r_y = np.random.uniform(-125.0, 129.5, size=trial)
    r_z = np.random.uniform(-117.5, -25, size=trial)
    random = np.stack((r_x, r_y, r_z), axis=1)

    M_xy = np.zeros((trial, trial))
    M_z = np.zeros((trial, trial))
    P_xy = np.zeros((trial, trial))
    P_z = np.zeros((trial, trial))
    Rand = np.random.uniform(size=(trial, trial))
    for i in range(0, trial):
        for j in range(0, trial):
            M_xy[i, j] = np.sqrt((random[i, 0] - random[j, 0]) ** 2 + (random[i, 1] - random[j, 1]) ** 2)
    for i in range(0, trial):
        for j in range(0, trial):
            M_z[i, j] = np.sqrt((random[i, 2] - random[j, 2]) ** 2)
    for i in range(0, trial):
        for j in range(0, trial):
            if Rand[i, j] <= xy_gaussian(M_xy[i, j]):
                P_xy[i, j] = 1
    for i in range(0, trial):
        for j in range(0, trial):
            if Rand[i, j] <= z_gaussian(M_xy[i, j]):
                P_z[i, j] = 1
    C = P_xy * P_z
    #     per = list(product(np.arange(500), repeat=3))

    #     for i in range(0, 2000):
    #         for j in range(0, 2000):
    #             for k in range(0, 2000):
    #                 if i!=j and j!=k and i!=k:
    #                     print('working?')
    #                     d = C[per[i][0:2]] + C[per[i][1:3]] + C[per[i][0], per[i][2]]
    #                             if d == 0:

    zero = 0
    one = 0
    two = 0
    three = 0
    for i in range(0, trial):
        for j in range(0, trial):
            for k in range(0, trial):
                if i != j and j != k and i != k:
                    if M_xy[i, j] <= 180 and M_z[i, j] <= 50 and M_xy[i, k] <= 180 and M_z[i, k] <= 180 and M_xy[
                        j, k] <= 180 and M_z[j, k] <= 50:
                        d = C[i, j] + C[i, k] + C[j, k]
                        if d == 0:
                            zero += 1
                        elif d == 1:
                            one += 1
                        elif d == 2:
                            two += 1
                        elif d == 3:
                            three += 1
    total = zero + one + two + three
    zero_series.append((i, zero / total))
    one_series.append((i, one / total))
    two_series.append((i, two / total))
    three_series.append((i, three / total))
    print('{} done'.format(i))

Parallel(n_jobs=-1)(delayed(main)(i) for i in r)
# print(num)
# num -= 1
np.savetxt('zero_series.txt', zero_series)
np.savetxt('one_series.txt', one_series)
np.savetxt('two_series.txt', two_series)
np.savetxt('three_series.txt', three_series)