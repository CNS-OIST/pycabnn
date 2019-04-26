import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

MLI = np.loadtxt('interneurons.txt')
MLI=MLI*0.5

trigger=len(MLI)
print(trigger)
random=MLI
dxy_M = np.zeros((trigger, trigger))
dz_M = np.zeros((trigger, trigger))
for i in range(0, trigger):
    for j in range(0, trigger):
        dxy_M[i, j] = np.sqrt((random[i, 0]-random[j, 0])**2 + (random[i, 1]-random[j, 1])**2)
        dz_M[i, j] = np.sqrt((random[i, 2]-random[j, 2])**2)
print('Complete distance matrix')

arr = dxy_M
mean = np.mean(arr)
variance = np.var(arr)
sigma = np.sqrt(variance)
arr1 = dz_M
mean1 = np.mean(arr1)
variance1 = np.var(arr1)
sigma1 = np.sqrt(variance1)
def xy_new_gaussian(x):
#     arr = dxy_M
# plt.figure(1)
# plt.hist(arr)
# plt.xlim((np.min(arr), np.max(arr)))
    y =  mlab.normpdf(x, mean, sigma)*152
    return y
# x = np.linspace(np.min(arr), np.max(arr), 100)
# plt.plot(x, mlab.normpdf(x, mean, sigma)+0.7)

# plt.show()
def z_new_gaussian(x):
#     arr1 = dz_M
# plt.figure(1)
# plt.hist(arr1, normed= True)
# plt.xlim((np.min(arr1), np.max(arr1)))
    y = mlab.normpdf(x, mean1, sigma1)*48
    return y
for w in range(0, 5):
    P_xy = np.zeros((trigger, trigger))
    P_z = np.zeros((trigger, trigger))
    Rand = np.random.uniform(size = (trigger, trigger))
    Rand1 = np.random.uniform(size = (trigger, trigger))

    for i in range(0, trigger):
        print(w, i)
        for j in range(0, trigger):
            if Rand[i, j] <= xy_new_gaussian(dxy_M[i, j]):
                P_xy[i, j] = 1
            if Rand1[i, j] <= z_new_gaussian(dz_M[i, j]):
                P_z[i, j] = 1
    print('Complete make probability matrix')

    zero = 0
    one = 0
    two = 0
    three = 0
    for i in range(0, trigger):
        print(w, i)
        for j in range(0, trigger):
            for k in range(0, trigger):
                if i != j and i != k and j != k:
                    d = P_xy[i, j]*P_z[i, j] + P_xy[i, k]*P_z[i, k] + P_xy[j, k]*P_z[j, k]
                    if d == 0:
                        zero += 1
                    elif d == 1:
                        one += 1
                    elif d == 2:
                        two += 1
                    elif d == 3:
                        three += 1
    total = zero + one + two + three
    zero_series_real.append((w, trigger, zero/total))
    one_series_real.append((w, trigger, one/total))
    two_series_real.append((w, trigger, two/total))
    three_series_real.append((w, trigger, three/total))

    print('Complete counting d')

    total = np.stack((zero, one, two, three))
    np.savetxt('zero_series_real_revised.txt', zero_series_real)
    np.savetxt('one_series_real_revised.txt', one_series_real)
    np.savetxt('two_series_real_revised.txt', two_series_real)
    np.savetxt('three_series_real_revised.txt', three_series_real)
    np.savetxt('total_revised.txt', total)