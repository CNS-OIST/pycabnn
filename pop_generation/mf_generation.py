import numpy as np
from scipy import spatial
from functools import partial
from tqdm import tqdm

# References: Fast Poisson Disk Sampling in Arbitrary Dimensions
#             Robert Bridson, SIGGRAPH, 2007
# Previous points and the spacing
# Setting properties of iteration


def make_2d_grid(sizeI, cellsize):
    return np.mgrid[0:sizeI[0]:cellsize, 0:sizeI[1]:cellsize]


def make_3d_grid(sizeI, cellsize):
    return np.mgrid[0:sizeI[0]:cellsize,
                    0:sizeI[1]:cellsize,
                    0:sizeI[2]:cellsize]


def Bridson_sampling_1(fgrid, sizeI, spacing, nPts, showIter):
    ndim = len(sizeI)
    cellsize = spacing / np.sqrt(ndim)
    # k = 5
    dartFactor = 4

    # Make grid size such that there is just one pt in each grid
    dm = spacing / np.sqrt(ndim)

    # Make a grid and convert it into a nx3 array
    # sGrid_nd = np.mgrid[0:sizeI[i]:cellsize for i in range(ndim)]
    sGrid_nd = fgrid(sizeI, cellsize)

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
    pbar = tqdm(total=nPts)
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

        bad_dart_grid = p[eligiblePts == False]
        scoreGrid[bad_dart_grid] = scoreGrid[bad_dart_grid] + 1
        # Update emptyGrid if scoreGrid has exceeded k dart throws

        # Update quantities for next iterations
        nEmptyGrid = emptyGrid.sum()
        pts = np.vstack((pts, tempPts))
        pts_newly_created = pts.shape[0] - ptsCreated
        ptsCreated = pts.shape[0]
        if showIter:
            # print('Iteration: {}    Points Created: {}    EmptyGrid:{}'.format(iter, pts.shape[0], nEmptyGrid))
            pbar.update(pts_newly_created)

        iter += 1
    # Cut down pts if more points are generated
    if pts.shape[0] > nPts:
        p = np.arange(pts.shape[0])
        p = np.random.choice(p, nPts, replace=False)
        pts = pts[p, :]

    if showIter:
        pbar.close()
        print('Iteration: {}    (final)Points Created: {}    EmptyGrid:{}'.format(iter, pts.shape[0], nEmptyGrid))

    return pts



Bridson_sampling_2d = partial(Bridson_sampling_1,
                              make_2d_grid)

Bridson_sampling_first = partial(Bridson_sampling_1,
                                 make_3d_grid)
