import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pts1 = np.loadtxt('Goc1.txt')
#pts2 = np.loadtxt('Glo1.txt')
pts3 = np.loadtxt('delete.txt')

#final = np.append(pts1, pts2, pts3)
#np.savetxt('Final.txt', final)
## Plot Setting

X1 = [x for (x, y, z) in pts1]
Y1 = [y for (x, y, z) in pts1]
#Z1 = [z for (x, y, z) in pts1]

#X2 = [x for (x, y, z) in pts2]
#Y2 = [y for (x, y, z) in pts2]
#Z2 = [z for (x, y, z) in pts2]

X3 = [x for (x, y, z) in pts3]
Y3 = [y for (x, y, z) in pts3]
#Z3 = [z for (x, y, z) in pts3]

fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1, projection='3d')
# ax1.scatter(X1, Y1, Z1, s=10, c = 'b')
# ax1.scatter(X2, Y2, Z2, s=10, c= 'r')
# ax1.scatter(X3, Y3, Z3, s=10, c = 'y')
ax1 = fig.add_subplot(111)
ax1.scatter(X1, Y1, s = 10, c='b')
#ax1.scatter(X2, Y2, s= 10, c = 'r')
ax1.scatter(X3, Y3, s = 10, c = 'y')
ax1.set_xlim(0, 100)
ax1.set_ylim(150, 250)
plt.show()