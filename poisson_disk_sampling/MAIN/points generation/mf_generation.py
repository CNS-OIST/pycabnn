# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
from scipy import spatial

# 2D version

def Bridson_sampling_mf(sizeI, spacing, nPts, showIter):
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007
    # Previous points and the spacing
    # Setting properties of iteration
    ndim = len(sizeI)
    cellsize = spacing / np.sqrt(ndim)
    k = 5
    dartFactor = 4

    # Make grid size such that there is just one pt in each grid
    dm = spacing / np.sqrt(ndim)

    # Make a grid and convert it into a nx3 array
    # sGrid_nd = np.mgrid[0:sizeI[0]:cellsize, 0:sizeI[1]:cellsize, 0:sizeI[2]:cellsize]
    # sGrid_nd = np.mgrid[0:sizeI[i]:cellsize for i in range(ndim)]
    sGrid_nd = np.mgrid[0:sizeI[0]:cellsize, 0:sizeI[1]:cellsize]
    # patchwork
    # sGrid_nd_1 =

    sGrid = np.array([sGrid_nd[i][:].flatten() for i in range(ndim)]).T
    del (sGrid_nd)
    # sizeGrid = np.size(sGrid[0])

    # Thrown in a particular grid
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
        availGrid, = np.where(emptyGrid)
        dataPts = min(nEmptyGrid, ndarts)
        p = np.random.choice(availGrid, int(dataPts), replace=False)
        tempPts = sGrid[p, :] + dm * np.random.rand(len(p), ndim)

        # Find good dart throws
        D, _ = spatial.cKDTree(np.vstack((pts, tempPts))).query(tempPts, k=2)
        # D = np.reshape(D[:, 1], (-1, 1))
        D = D[:, 1]

        #Dist1, _ = spatial.cKDTree(np.vstack((pts2, tempPts))).query(tempPts, k=2)
        #Dist1 = Dist1[:, 1]

        withinI = np.array([tempPts[:, i] < sizeI[i] for i in range(ndim)]).T
        withinI = np.array([np.prod(x) for x in withinI])
        #eligiblePts = (withinI>0)*(D>spacing)*(Dist>p_spacing)*(Dist1 > p_spacing1)
        eligiblePts = (withinI > 0) * (D > spacing)
        # scorePts = tempPts[eligiblePts==False, :]
        tempPts = tempPts[eligiblePts, :]
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
        bad_dart_grid = p[eligiblePts == False]
        scoreGrid[bad_dart_grid] = scoreGrid[bad_dart_grid] + 1
        # Update emptyGrid if scoreGrid has exceeded k dart throws

        # Update quantities for next iterations
        nEmptyGrid = emptyGrid.sum()
        pts = np.vstack((pts, tempPts))
        ptsCreated = pts.shape[0]
        if showIter:
            print('Iteration: {}    Points Created: {}    EmptyGrid:{}'.format(iter, pts.shape[0], nEmptyGrid))
        iter += 1
    # Cut down pts if more points are generated
    if pts.shape[0] > nPts:
        p = np.arange(pts.shape[0])
        p = np.random.choice(p, nPts, replace=False)
        pts = pts[p, :]

    if showIter:
        print('Iteration: {}    (final)Points Created: {}    EmptyGrid:{}'.format(iter, pts.shape[0], nEmptyGrid))

    return pts


# if __name__ == '__main__':
#     Longaxis = 1500
#     Shortaxis = 700
#     height = 200
#     numMF = 200000
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1, 1, 1, projection = '3d')
#     #pts1 = np.loadtxt('test1.txt')
#     #pts2 = np.loadtxt('Glo1.txt')
#     points1 = Bridson_sampling((Longaxis, Shortaxis, height), 10, int(numMF), True)
#     X = (x for (x, y, z) in points1)
#     Y = (y for (x, y, z) in points1)
#     Z = (z for (x, y, z) in points1
#     ax1.scatter(X, Y, Z, s=10)
#     add_noise = points1 + np.random.normal(scale = sigma/np.sqrt(3), size = (len(points1), 3))


#     np.savetxt('test2.txt', points1)
#
#     X1 = [x for (x, y) in points1]
#     Y1 = [y for (x, y) in points1]
#     X3 = [x for (x, y) in pts1]
#     Y3 = [y for (x, y) in pts1]
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax1.scatter(X1, Y1, s = 10, c='b')
#     #ax1.scatter(X2, Y2, s= 10, c = 'r')
#     ax1.scatter(X3, Y3, s = 10, c = 'y')
#     ax1.set_xlim(0, 300)
#     ax1.set_ylim(0, 300)
#     plt.show()


#     #     pts1 = points1[1]
#     #     #####Figure_Section#####
#     #     plt.close('all')
#     fig = plt.figure()
#     # subplot setting
#     # ax1 = fig.add_subplot(1, 1, 1, projection='3d')
#     ax1 = fig.add_subplot(1, 1, 1)
#     # fig.subplots_adjust(hspace= 0.5, wspace = 0.3)
#     X = [x for (x, y) in points1]
#     Y = [y for (x, y) in points1]
#     # Z = [z for (x, y, z) in points1[0]]
#     #
#     #     X1 = [x for (x, y) in pts1]
#     #     Y1 = [y for (x, y) in pts1]
#     #     #Z1 = [z for (x, y, z) in pts1]
#     #     #Another things..
#     #     #ax1.scatter(X, Y, Z, s= 10)
#     #     #ax1.scatter(X1, Y1, Z1, s=100, c='r')
#     ax1.scatter(X, Y, s=10)
# #     ax1.scatter(X1, Y1, s=10, c='r')
# #     ax1.set_xlim(0, 1)
# #     ax1.set_ylim(0, 1)
# #     #ax1.set_zlim(0, 1)
# #     ax1.set_xlabel('X')
# #     ax1.set_ylabel('Y')
# #     #ax1.set_zlabel('Z')
# #
# plt.show()
