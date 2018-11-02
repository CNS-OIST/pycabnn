# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
# 3D version

def Bridson_sampling(sizel=(2, 2, 2), radius=0.005, k=42874):
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007

    # Here `3` corresponds to the number of dimension
    ndim = len(sizel)
    cellsize = radius/np.sqrt(ndim)
    k = 5
    dartFactor = 4
    #rows = int(np.ceil(width/cellsize))
    #cols = int(np.ceil(depth/cellsize))
    #third = int(np.ceil(height/cellsize))

    #Make a grid
    for i in range(1, 3):
        Grid[i] = np.array(np.mgrid([0:sizel[i-1]))

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
        return 0 <= p[0] < width and 0 <= p[1] < depth and 0 <= p[2] < height

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
            if M[i, j, u] and squared_distance(p, P[i, j, u]) < cubic_radius :
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
    #theta = []
    while (count < k):
        #if theta < 0.8:
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
                    #dx = np.diff(count_count)
                    #dy = np.diff(reject_count)
                    #theta = np.arctan(dy/dx)
       # else:
            #break
    return P[M], count_count, reject_count, count_time, elapsed_time, width, depth, height, count