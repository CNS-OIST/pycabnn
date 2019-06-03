"""
References: Fast Poisson Disk Sampling in Arbitrary Dimensions
            Robert Bridson, SIGGRAPH, 2007
Previous points and the spacing
Setting properties of iteration
"""

import numpy as np

# from scipy import spatial
from functools import partial
from tqdm.autonotebook import tqdm
from sklearn.neighbors import KDTree


def make_2d_grid(sizeI, cellsize):
    return np.mgrid[0 : sizeI[0] : cellsize, 0 : sizeI[1] : cellsize]


def make_3d_grid(sizeI, cellsize):
    return np.mgrid[
        0 : sizeI[0] : cellsize, 0 : sizeI[1] : cellsize, 0 : sizeI[2] : cellsize
    ]


def set_nDarts(nPts, nEmptyGrid, dartFactor=4):
    # if nPts == 0:
    #     nPts = nEmptyGrid
    ndarts = np.round(nEmptyGrid / dartFactor)
    # ndarts = np.round(nPts / dartFactor)
    return int(ndarts)


def Bridson_sampling_1(fgrid, sizeI, spacing, nPts, showIter, discount_factor=0.5):
    ndim = len(sizeI)
    cellsize = spacing / np.sqrt(ndim)
    # k = 5

    # Make grid size such that there is just one pt in each grid
    dgrid = spacing / np.sqrt(ndim)

    # Make a grid and convert it into a nxD array
    sGrid_nd = fgrid(sizeI, cellsize)

    sGrid = np.array([sGrid_nd[i][:].flatten() for i in range(ndim)]).T
    del sGrid_nd

    # Thrown in a particular grid
    is_grid_empty = np.ones(sGrid.shape[0], dtype=bool)
    nEmptyGrid = is_grid_empty.sum()
    scoreGrid = np.ones(sGrid.shape[0])
    scoreGrid = scoreGrid

    # Initialize Parameters
    if nPts == 0:
        nPts = nEmptyGrid
    n_pts_created = 0
    pts = np.empty(shape=(1, ndim))
    iter = 0

    # Start Iterative process
    pbar = tqdm(total=nPts)

    while n_pts_created < nPts and nEmptyGrid > 0:
        # Thrown darts in eligible grids
        availGrid, = np.where(is_grid_empty)
        score_availGrid = scoreGrid[availGrid]
        score_availGrid = score_availGrid / score_availGrid.sum()

        ndarts = set_nDarts(nPts, nEmptyGrid)
        p = np.random.choice(availGrid, ndarts, replace=False, p=score_availGrid)
        tempPts = sGrid[p, :] + dgrid * np.random.rand(len(p), ndim)

        # Find good dart throws
        D, _ = KDTree(np.vstack((pts, tempPts))).query(tempPts, k=2)
        D = D[:, 1]

        # withinI = np.array([tempPts[:, i] < sizeI[i] for i in range(ndim)]).T
        # withinI = np.array([np.prod(x) for x in withinI])
        # is_eligible = (withinI > 0) * (D > spacing)
        is_eligible = D > spacing

        accepted_pts = tempPts[is_eligible, :]

        accepted_grids = p[is_eligible]
        rejected_grids = p[~is_eligible]

        is_grid_empty[accepted_grids] = False

        scoreGrid[rejected_grids] = scoreGrid[rejected_grids] * discount_factor

        # Update quantities for next iterations
        nEmptyGrid = is_grid_empty.sum()
        pts = np.vstack((pts, accepted_pts))
        n_pts_newly_created = pts.shape[0] - n_pts_created
        n_pts_created = pts.shape[0]
        if showIter:
            pbar.update(n_pts_newly_created)

        iter += 1
    # Cut down pts if more points are generated
    if pts.shape[0] > nPts:
        p = np.arange(pts.shape[0])
        p = np.random.choice(p, nPts, replace=False)
        pts = pts[p, :]

    if showIter:
        pbar.close()
        print(
            "Iteration: {}, (final)Points Created: {}, is_grid_empty:{} ({}%)".format(
                iter, pts.shape[0], nEmptyGrid, nEmptyGrid / sGrid.shape[0] * 100
            )
        )

    return pts


Bridson_sampling_2d = partial(Bridson_sampling_1, make_2d_grid)

Bridson_sampling_first = partial(Bridson_sampling_1, make_3d_grid)

