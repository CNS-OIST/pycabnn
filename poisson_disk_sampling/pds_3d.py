# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# 3D version

def Bridson_sampling(width=0.1, depth=0.1, height=0.1, radius=0.005, k=43000):
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007

    # Here `3` corresponds to the number of dimension
    cellsize = radius/np.sqrt(3)
    rows = int(np.ceil(width/cellsize))
    cols = int(np.ceil(depth/cellsize))
    third = int(np.ceil(height/cellsize))

    # Positions cells
    P = np.zeros((rows, cols, third, 3), dtype=np.float32)
    M = np.zeros((rows, cols, third), dtype=bool)

    # Cubic radius because we'll compare cubic distance
    cubic_radius = radius*radius*radius

    def squared_distance(p0, p1):
        return (p0[0]-p1[0])**2 + (p0[1]-p1[1])**2 + (p0[2]-p1[2])**2

    def random_point_around(p, k=1):
        # WARNING: This is not uniform around p but we can live with it
        R = np.random.uniform(radius, 2*radius, k)
        T = np.random.uniform(0, 2*np.pi, k)
        O = np.random.uniform(0, 2 * np.pi, k)
        P = np.empty((k, 3))
        P[:, 0] = p[0]+R*np.cos(T)*np.sin(O) #x axis
        P[:, 1] = p[1]+R*np.sin(T)*np.sin(O) #y axis
        P[:, 2] = p[2] + R*np.cos(T) # z axis
        return P

    def in_limits(p):
        return 0 <= p[0] < width and 0 <= p[1] < depth and 0<= p[2] < height

    def neighborhood(shape, index, n):
        row, col, third = index
        row0, row1 = max(row-n, 0), min(row+n+1, shape[0])
        col0, col1 = max(col-n, 0), min(col+n+1, shape[1])
        third0, third1 = max(third-n, 0), min(third+n+1, shape[2])
        I = np.stack(np.mgrid[row0:row1, col0:col1, third0:third1], axis=3)
        I = I.reshape(I.size//3, 3).tolist()
        I.remove([row, col, third])
        return I

    def in_neighborhood(p):
        i, j, u = int(p[0]/cellsize), int(p[1]/cellsize), int(p[2]/cellsize)
        if M[i, j, u]:
            return True
        for (i, j, u) in N[(i, j, u)]:
            if M[i, j, u] and squared_distance(p, P[i, j, u]) < cubic_radius : #I changed cubic radius just radius.
                return True
        return False

    def add_point(p):
        points.append(p)
        i, j, u = int(p[0]/cellsize), int(p[1]/cellsize), int(p[2]/cellsize)
        P[i, j, u], M[i, j, u] = p, True

    # Cache generation for neighborhood
    N = {}
    for i in range(rows):
        for j in range(cols):
            for u in range(third):
                N[(i, j, u)] = neighborhood(M.shape, (i, j, u), 3)

    #Main process
    points = []
    add_point((width/2, depth/2, height/2))
    count = 1
    reject = 0
    reject_count = []
    count_count = []
    elapsed_time = []
    count_time = []
    while (count < k):
        l = np.random.randint(count)
        p = points[l]
        Q = random_point_around(p, 30)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q):
                add_point(q)
                # t1_end = time.perf_counter()
                print(count)
                print(q)
                count += 1
                end_time = time.perf_counter()
                # t1_start = time.perf_counter()
                elapsed_time.append(np.abs(end_time - start_time))
                count_time.append(count)
            else:
                reject += 1
                reject_count.append(reject)
                count_count.append(count)

    return P[M], count_count, reject_count, count_time, elapsed_time


if __name__ == '__main__':
    start_time = time.perf_counter()
    points1 = Bridson_sampling()
    count_count = points1[1]
    reject_count = points1[2]
    count_time = points1[3]
    elapsed_time = points1[4]

    plt.close('all')
    fig = plt.figure()
    #subplot setting
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 4)
    #fig.subplots_adjust(hspace= 0.5, wspace = 0.3)
    X = [x for (x, y, z) in points1[0]]
    Y = [y for (x, y, z) in points1[0]]
    Z = [z for (x, y, z) in points1[0]]
    #subplot data
    X1 = count_count
    Y1 = reject_count
    X2 = count_time
    Y2 = elapsed_time
    #Another things..
    ax1.scatter(X, Y, Z, s= 10)
    ax1.set_xlim(0, 0.1)
    ax1.set_ylim(0, 0.1)
    ax1.set_zlim(0, 0.1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2.plot(X1, Y1)
    ax2.set_xlabel('Number of Previous count')
    ax2.set_ylabel('Number of Rejection')

    ax3.plot(Y2, X2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Number of Count')

    plt.tight_layout()
    plt.savefig('3D poisson-disk-sampling.png')
    plt.show()
