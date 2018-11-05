# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
from scipy import spatial

# 3D version

def Bridson_sampling(sizeI=(2, 2, 2), radius=0.005, showIter=0):
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007

    #Parsing inputs and setting default values
    if nargin == 3:
        showIter = 0
    if nargin == 2:
        showIter = 0
        nPts = 0

    #Setting properties of iteration
    ndim = len(sizeI)
    cellsize = radius / np.sqrt(ndim)
    k = 5
    dartFactor = 4

    #Make grid size such that there is just one pt in each grid
    dm = radius/sqrt(dim)

    # Make a grid and convert it into a nx3 array 계속 지켜봐야함
    sGrid = np.mgrid[row0:row1:cellsize, col0:col1:cellsize, third0:third1:cellsize]
    for i in range(1, ndim):
        sGrid[i] = np.reshape(sGrid[i][:], (-1, 1))
    sizeGrid = np.size(sGrid[0])

    #Thrown in a particular grid
    emptyGrid = np.ones(Grid, dtype=bool)
    nEmptyGrid = sum(emptyGrid)
    scoreGrid = np.zeros(Grid)

    # Darts to be thrown per iterations
    if nPts == 0:
        nPts = nEmptyGrid
        ndarts = round(nEmptyGrid / dartFactor)
    ndarts = round(nPts / dartFactor)

    # Initialize Parameters
    ptsCreated = 0
    pts = []
    iter = 0

    # Start Iterative process
    while ptsCreated < nPts and nEmptyGrid > 0:
        # Thrown darts in eligible grids
        availGrid = np.where(emptyGrid == 1)
        dataPts = min(nEmptyGrid, ndarts)
        p = np.random.choice(availGrid, dataPts, replace=False)
        tempPts = sGrid[p, :] + dm * rand(len(p), ndim)

        # Find good dart throws
        tree = spatial.KDtree([np.vstack(pts, tempPts)], tempPts, k=2)
        tree = tree[:, 1]

        withinI = logical(2 * (min(tempPts, sizel)))  # 이거 여쭈어보기
        eligiblePts = withinI and tree > spacing

        scorePts = tempPts[~eligiblePts, :]
        tempPts = temepPts[eligiblePts, :]

        # Update empty Grid
        emptyPts = np.floor((tempPts + dm - 1) / dm)
        emptyPts_0 = emptyPts[0]
        emptyPts_1 = emptyPts[1]
        emptyPts_2 = emptyPts[2]
        emptyPts = (emptyPts_0, emptyPts_1, emptyPts_2)
        emptyIdx = np.ravel(sizeGrid, emptyPts)  # 이거 좀 더 유심히
        emptyGrid[emptyIdx] = 0

        # Update score pts
        scorePts = floor((scorePts + dm - 1) / dm)
        scorePts_0 = scorePts[0]
        scorePts_1 = scorePts[1]
        scorePts_2 = scorePts[2]
        scorePts = [scorePts_0, scorePts_1, scorePts_2]
        scoreIdx = np.ravel(sizeGrid, scorePts)
        scoreGrid[scoreIdx]= scoreGrid([scoreIdx] + 1

        # Update emptyGrid if scoreGrid has exceeded k dart throws
        emptyGrid1 = np.intersect1d(emptyGrid, (scoreGrid < k)) #왜 에러가 날까?

        # Update quantities for next iterations
        nEmptyGrid = sum(emptyGrid1)
        pts = np.vstack(pts, tempPts)
        ptsCreated = size(pts, 1)
        iter += 1
    end

    # Cut down pts if more points are generated
    if np.size(pts, 1) > nPts:
        p = np.arange(1, size(pts, 1)+1)
        p = np.random.choice(p, nPts, replace=False)
        pts = pts[p, :]

    return pts
    if __name__ == '__main__':
        points1 = Bridson_sampling()
    print(points1)