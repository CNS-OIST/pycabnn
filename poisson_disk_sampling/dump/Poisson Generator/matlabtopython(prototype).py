# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D version

def Bridson_sampling(sizeI=(1, 1, 1), spacing=0.005, nPts=100000, showIter=True):
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007

    #Setting properties of iteration
    ndim = len(sizeI)
    cellsize = spacing / np.sqrt(ndim)
    k = 5
    dartFactor = 4

    #Make grid size such that there is just one pt in each grid
    dm = spacing/np.sqrt(ndim)

    # Make a grid and convert it into a nx3 array 돌려봐야함
    sGrid_3d = np.mgrid[0:sizeI[0]:cellsize, 0:sizeI[1]:cellsize, 0:sizeI[2]:cellsize]

    sGrid = np.array([sGrid_3d[i][:].flatten() for i in range(ndim)]).T
    del(sGrid_3d)
    sizeGrid = np.size(sGrid[0])
    print('A grid is successfully created with size {}'.format(sizeGrid))

    #Thrown in a particular grid
    emptyGrid = np.ones(sGrid.shape[0], dtype=bool)
    nEmptyGrid = emptyGrid.sum()
    scoreGrid = np.zeros(sGrid.shape[0])

    # Darts to be thrown per iterations
    if nPts == 0:
        nPts = nEmptyGrid
        ndarts = np.round(nEmptyGrid / dartFactor)
    ndarts = np.round(nPts / dartFactor)

    # Initialize Parameters
    ptsCreated = 0
    pts = np.empty(shape=(1, ndim))
    iter = 0

    # Start Iterative process
    while ptsCreated < nPts and nEmptyGrid > 0:
        # Thrown darts in eligible grids
        availGrid,  = np.where(emptyGrid)
        dataPts = min(nEmptyGrid, ndarts)
        p = np.random.choice(availGrid, int(dataPts), replace=False)
        print('Throw darts at {} points'.format(p.size))
        tempPts = sGrid[p, :] + dm * np.random.rand(len(p), ndim)
        print('Dart positions generated.')

        # Find good dart throws
        D, _ = spatial.KDTree(np.vstack((pts, tempPts))).query(tempPts, k=2)
        #D = np.reshape(D[:, 1], (-1, 1))
        D = D[:, 1]
        print('KDTree search complete')

        withinI = np.array([tempPts[:, i] < sizeI[i] for i in range(ndim)]).T
        withinI = np.array([np.prod(x) for x in withinI])
        eligiblePts = (withinI>0)*(D>spacing)
        print('eligiblePTs is generated')
       # scorePts = tempPts[eligiblePts==False, :]
        tempPts = tempPts[eligiblePts, :]
        print('tempPTs is generated')
        # Update empty Grid
    #     emptyPts = np.floor((tempPts + dm - 1) / dm)
    #     emptyPts_0 = emptyPts[0]
    #     emptyPts_1 = emptyPts[1]
    #     emptyPts_2 = emptyPts[2]
    #     emptyPts = (emptyPts_0, emptyPts_1, emptyPts_2)
    #     emptyIdx = np.ravel(sizeGrid, emptyPts)  # 이거 좀 더 유심히
    #     emptyGrid[emptyIdx] = 0
        # alternatively,
        emptyGrid[p[eligiblePts]] = False

    #     # Update score pts
    #     scorePts = floor((scorePts + dm - 1) / dm)
    #     scorePts_0 = scorePts[0]
    #     scorePts_1 = scorePts[1]
    #     scorePts_2 = scorePts[2]
    #     scorePts = [scorePts_0, scorePts_1, scorePts_2]
    #     scoreIdx = np.ravel(sizeGrid, scorePts)
    #     scoreGrid[scoreIdx]= scoreGrid([scoreIdx] + 1
    #
        bad_dart_grid = p[eligiblePts==False]
        scoreGrid[bad_dart_grid] = scoreGrid[bad_dart_grid] + 1
        print('ScoreGrid update complete')
    # Update emptyGrid if scoreGrid has exceeded k dart throws
        emptyGrid = emptyGrid*(scoreGrid < k)
        print('EmptyGrid update complete')
        # Update quantities for next iterations
        nEmptyGrid = emptyGrid.sum()
        print('nEmptyGrid update complete')
        pts = np.vstack((pts, tempPts))
        print('pts update is completed')
        ptsCreated = pts.shape[0]
        print('ptsCreated is completed')
        iter += 1
        print('Each parameters for next iteration is completed')
    # Cut down pts if more points are generated
    if np.size(pts, 1) > nPts:
        p = np.arange(pts.shape[0])
        p = np.random.choice(p, nPts, replace=False)
        pts = pts[p, :]

    if showIter:
        print('Iteration: {}    Points Created: {}    EmptyGrid:{}'.format(iter,ptsCreated,nEmptyGrid))

    return pts

if __name__ == '__main__':
    points1 = Bridson_sampling()

    #####Figure_Section#####
    plt.close('all')
    fig = plt.figure()
    #subplot setting
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    #fig.subplots_adjust(hspace= 0.5, wspace = 0.3)
    X = [x for (x, y, z) in points1]
    Y = [y for (x, y, z) in points1]
    Z = [z for (x, y, z) in points1]
    #Another things..
    ax1.scatter(X, Y, Z, s= 10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    plt.show()