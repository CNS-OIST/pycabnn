# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
from scipy import spatial
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import pickle
#import time
# 3D version

def Bridson_sampling_3(sizeI, spacing, nPts, showIter, pts1, pts2):
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007
    # Previous points and the spacing
    #pts1 = np.random.rand(30, 3) #Should be input value of this function
    count = 0
    count_time = []
    elapsed_time= []
    #Setting properties of iterati
    ndim = len(sizeI)
    cellsize = spacing / np.sqrt(ndim)
    k = 5
    dartFactor = 4

    #Make grid size such that there is just one pt in each grid
    dm = spacing/np.sqrt(ndim)

    # Make a grid and convert it into a nx3 array
    sGrid_nd = np.mgrid[0:sizeI[0]:cellsize, 0:sizeI[1]:cellsize, 0:sizeI[2]:cellsize]
    #sGrid_nd = np.mgrid[0:sizeI[i]:cellsize for i in range(ndim)]

    #patchwork
    #sGrid_nd_1 =

    sGrid = np.array([sGrid_nd[i][:].flatten() for i in range(ndim)]).T
    del(sGrid_nd)
    #sizeGrid = np.size(sGrid[0])

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
    itercount = np.zeros(shape=(10000, 2))
    # Start Iterative process
    while ptsCreated < nPts and nEmptyGrid > 0:
        # Thrown darts in eligible grids
        availGrid,  = np.where(emptyGrid)
        dataPts = min(nEmptyGrid, ndarts)
        p = np.random.choice(availGrid, int(dataPts), replace=False)
        tempPts = sGrid[p, :] + dm * np.random.rand(len(p), ndim)

        # Find good dart throws
        D, _ = spatial.cKDTree(np.vstack((pts2, pts1, pts, tempPts))).query(tempPts, k=2)
        #D = np.reshape(D[:, 1], (-1, 1))
        D = D[:, 1]

        #Check with previous points
        Dist, _ = spatial.KDTree(np.vstack((pts1, tempPts))).query(tempPts, k=2) #This is a gap between goc
        Dist = Dist[:, 1]

        #Check with previous points (one more!)
        Dist1, _ = spatial.KDTree(np.vstack((pts2, tempPts))).query(tempPts, k=2) #This is a gap between glo
        Dist1 = Dist1[:, 1]


        withinI = np.array([tempPts[:, i] < sizeI[i] for i in range(ndim)]).T
        withinI = np.array([np.prod(x) for x in withinI])
        eligiblePts = (withinI>0)*(D>spacing)*(Dist > 10)*(Dist1> 6)
        #(Dist > p_spacing) * (Dist1 > p_spacing1)

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
        bad_dart_grid = p[eligiblePts==False]
        scoreGrid[bad_dart_grid] = scoreGrid[bad_dart_grid] + 1
        # Update emptyGrid if scoreGrid has exceeded k dart throws

        # Update quantities for next iterations
        nEmptyGrid = emptyGrid.sum()
        pts = np.vstack((pts, tempPts))
        ptsCreated = pts.shape[0]
        if showIter:
            print('Iteration: {}    Points Created: {}    EmptyGrid:{}'.format(iter, pts.shape[0], nEmptyGrid))
        iter += 1
        itercount[iter, 0] = iter
        itercount[iter, 1] = ptsCreated
        iter += 1
        if itercount[iter - 1, 1] == itercount[iter - 5, 1]:
            print('I checked it')
            break
        # count_time.append(count)
        # end_time = time.perf_counter()
        # elapsed_time.append(np.abs(end_time - start_time))
        count += 1
    # Cut down pts if more points are generated
    if pts.shape[0] > nPts:
        p = np.arange(pts.shape[0])
        p = np.random.choice(p, nPts, replace=False)
        pts = pts[p, :]


    if showIter:
        print('Iteration: {}    (final)Points Created: {}    EmptyGrid:{}'.format(iter,pts.shape[0],nEmptyGrid))
    return pts