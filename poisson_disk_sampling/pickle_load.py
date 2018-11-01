import pickle
import matplotlib.pyplot as plt

pickle_off = open("[data]3D poisson disk sampling, width = 0.10, depth = 0.10, height=0.10, points = 20000, 2018-10-31_14-21-13", "rb")
emp = pickle.load(pickle_off)




## Plot Setting

X = [x for (x, y, z) in emp]
Y = [y for (x, y, z) in emp]
Z = [z for (x, y, z) in emp]

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
ax1.scatter(X, Y, Z, s=10)
ax1.set_xlim(0, 0.1)
ax1.set_ylim(0, 0.1)
ax1.set_zlim(0, 0.1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
plt.show()