# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import time

def Bridson_sampling(width=1.0, height=1.0, radius=0.005, k=2700):
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007

    # Here `2` corresponds to the number of dimension
    cellsize = radius/np.sqrt(2)
    rows = int(np.ceil(width/cellsize))
    #rows1 = rows/2
    #rows2 =
    cols = int(np.ceil(height/cellsize))

    # Squared radius because we'll compare squared distance
    squared_radius = radius*radius

    # Positions cells
    P = np.zeros((rows, cols, 2), dtype=np.float32)
    M = np.zeros((rows, cols), dtype=bool)

    def squared_distance(p0, p1):
        return (p0[0]-p1[0])**2 + (p0[1]-p1[1])**2

    def random_point_around(p, k=1):
        # WARNING: This is not uniform around p but we can live with it
        R = np.random.uniform(radius, 2*radius, k)
        T = np.random.uniform(0, 2*np.pi, k)
        P = np.empty((k, 2))
        P[:, 0] = p[0]+R*np.sin(T)
        P[:, 1] = p[1]+R*np.cos(T)
        return P

    def in_limits(p):
        return 0 <= p[0] < width and 0 <= p[1] < height

    def neighborhood(shape, index, n=2): #Limit the searching area around the dot which we want to point
        row, col = index
        row0, row1 = max(row-n, 0), min(row+n+1, shape[0])
        col0, col1 = max(col-n, 0), min(col+n+1, shape[1])
        I = np.stack(np.mgrid[row0:row1, col0:col1], axis=2)
        I = I.reshape(I.size//2, 2).tolist() #정확히 무슨 역할을 하는지 모르겠음
        I.remove([row, col]) #원래 자기가 있던 위치를 제거
        return I

    def in_neighborhood(p):
        i, j = int(p[0]/cellsize), int(p[1]/cellsize)
        if M[i, j]: # The grid which has the point (p) now.
            return True
        for (i, j) in N[(i, j)]: # Limit the area from the points to the neighborhood
            if M[i, j] and squared_distance(p, P[i, j]) < squared_radius: #It was squared radius but I changed it. If r>1, which is r is bigger than 100um, then r>r^2 but now, since r<1, r^2>r .
                return True
        return False

    def add_point(p):
        points.append(p)
        i, j = int(p[0]/cellsize), int(p[1]/cellsize)
        P[i, j], M[i, j] = p, True

    row_1 = int(rows/2)
    col_1 = int(cols/2)

    # Cache generation for neighborhood #[i, j]만 주어지면 그에 대한 주변 영역을 바로 서치할 수 있게 미리 모든 i, j에 대해 만들어놓음.
    N = {}
    for i in range(1, row_1):
        for j in range(1, col_1):
            N[(i, j)] = neighborhood(M.shape, (i, j), 2)  # So, the N has the informaiton of which grids are in near the point
    for i in range(int(rows/2)+1, rows):
        for j in range(int(cols/2)+1, cols):
            N[(i, j)] = neighborhood(M.shape, (i, j), 2)  # So, the N has the informaiton of which grids are in near the point

    # Main process
    points = []
    #add_point((np.random.uniform(width), np.random.uniform(height))) # The first point
    add_point((width/2, height/2))
    count = 1
    reject = 0
    reject_count = []
    count_count = []
    elapsed_time = []
    count_time = []
    while(count < k) :
        l = np.random.randint(count)
        p = points[l]
        Q = random_point_around(p, 30)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q):
                add_point(q)
                #t1_end = time.perf_counter()
                print(count)
                print(q)
                count += 1
                end_time = time.perf_counter()
                #t1_start = time.perf_counter()
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
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 4)
    #fig.subplots_adjust(hspace= 0.5, wspace = 0.3)
    X = [x for (x, y) in points1[0]]
    Y = [y for (x, y) in points1[0]]

    #subplot data
    X1 = count_count
    Y1 = reject_count
    X2 = count_time
    Y2 = elapsed_time

    #Another things..
    ax1.scatter(X, Y, s= 10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.plot(X1, Y1)
    ax3.plot(Y2, X2)
    ax1.set_xlabel('X', fontsize = 15)
    ax1.set_ylabel('Y', fontsize = 15)
    ax2.set_xlabel('Number of Previous count', fontsize = 15)
    ax2.set_ylabel('Number of Rejection', fontsize = 15)
    ax3.set_xlabel('Time', fontsize = 15)
    ax3.set_ylabel('Number of Count', fontsize = 15)
    plt.tight_layout()
    plt.savefig('3D poisson-disk-sampling.png')
    plt.show()