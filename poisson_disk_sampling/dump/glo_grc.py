import numpy as np
from joblib import Parallel, delayed
from scipy import spatial
import matplotlib.pyplot as plt

grc_points = np.loadtxt('noise_grc.txt')
points = np.loadtxt('noise_glo.txt')
total = np.vstack((grc_points, points))

def new_count_grc(x, total, r, p):
    temp=[]
    count=[]
    ball = spatial.cKDTree(total).query_ball_point(x, r)
    for results in ball: #Setting temporal coordinates which points in the circle
        temp.append(total[results])
    for i in range(0, np.size(x[:, 0])): #Go to each central point
        new=[]
        num = np.size(temp[i])//3 #Setting how many points around one center point(i)
        while num>0:
            if x[i, 1]-r/4 <= temp[i][num-1, 0] <= x[i, 1]+r/4 and x[i, 2]-15 <= temp[i][num-1, 2] <= x[i, 2]+15: # Restrict the x-radius and z-radius
                new.append(temp[i][num-1, :])
            num -= 1
        new = np.random.choice(new, size=len(new)*p, replace=False)
        count.append(len(new))
    return r, np.mean(count), np.var(count)

def new_count_glo(x, total, r):
    temp=[]
    count=[]
    ball = spatial.cKDTree(total).query_ball_point(x, r)
    for results in ball: #Setting temporal coordinates which points in the circle
        temp.append(total[results])
    for i in range(0, np.size(x[:, 0])): #Go to each central point
        new=[]
        num = np.size(temp[i])//3 #Setting how many points around one center point(i)
        while num>0:
            if x[i, 1]-r/3 <= temp[i][num-1, 0] <= x[i, 1]+r/3 and x[i, 2]-10 <= temp[i][num-1, 2] <= x[i, 2]+10: # Restrict the x-radius and z-radius
                new.append(temp[i][num-1, :])
            num -= 1
        count.append(len(new))
    return r, np.mean(count), np.var(count), p

rp = np.arange(0, 1, 0.3)
rs_grc = np.arange(20, 21, 0.5)
rs_glo = np.arange(25, 35, 0.1)
c_grc= rs_grc[0]
c_glo = rs_glo[0]

e1 = Parallel(n_jobs=-1)(delayed(new_count_grc)(grc_points, total, r, p) for r in rs_grc for p in rp)
e1_1 = np.reshape(e1, (-1, 4))
print(e1_1)
np.savetxt('center_grc.txt', e1_1)

e2 = Parallel(n_jobs=-1)(delayed(new_count_glo)(points, total, r) for r in rs_glo)
e2_1 = np.reshape(e2, (-1, 3))
np.savetxt('center_glo.txt', e2_1)