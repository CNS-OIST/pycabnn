# References: Fast Poisson Disk Sampling in Arbitrary Dimensions
#             Robert Bridson, SIGGRAPH, 2007

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


def set_nDarts(nPts, n_pts_created, nEmptyGrid, dartFactor=4):
    n_to_create = nPts - n_pts_created

    ndarts = np.min([np.round(nPts / dartFactor), nEmptyGrid])
    # ndarts = np.round(nPts / dartFactor)
    return int(np.round(ndarts))


def Bridson_sampling_1(
    fgrid, sizeI, spacing, nPts, showIter, ftests=[], discount_factor=0.75
):
    count = 0
    count_time = []
    elapsed_time = []

    # Setting properties of iterati
    ndim = len(sizeI)
    cellsize = spacing / np.sqrt(ndim)

    # Make grid size such that there is just one pt in each grid
    dgrid = spacing / np.sqrt(ndim)

    # Make a grid and convert it into a nxD array
    sGrid_nd = fgrid(sizeI, cellsize)

    sGrid = np.array([sGrid_nd[i][:].flatten() for i in range(ndim)]).T
    del sGrid_nd

    # Thrown in a particular grid
    nEmptyGrid = nEmptyGrid0 = sGrid.shape[0]
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
        # availGrid, = np.where(is_grid_empty)
        # score_availGrid = scoreGrid[availGrid]
        scoreGrid = scoreGrid / scoreGrid.sum()

        ndarts = set_nDarts(nPts, n_pts_created, nEmptyGrid)
        if ndarts != sGrid.shape[0]:
            p = np.random.choice(
                range(sGrid.shape[0]), ndarts, replace=False, p=scoreGrid
            )
            tempPts = sGrid[p, :] + dgrid * np.random.rand(len(p), ndim)
        else:
            tempPts = sGrid + dgrid * np.random.rand(len(ndarts), ndim)

        # withinI = np.array([tempPts[:, i] < sizeI[i] for i in range(ndim)]).T
        # withinI = np.array([np.prod(x) for x in withinI])
        # eligiblePts = (withinI>0)*(D>spacing)*(Dist > 10)

        is_safe_to_continue = 1

        if ftests != []:
            is_safe_with_prev_pts = np.ones(len(p), dtype=bool)
            for ftest in ftests:
                is_safe_with_prev_pts = is_safe_with_prev_pts * ftest(tempPts)

            is_safe_to_continue = np.sum(is_safe_with_prev_pts)
            rejected_grids = p[~is_safe_with_prev_pts]
            scoreGrid[rejected_grids] = scoreGrid[rejected_grids] * discount_factor

            p = p[is_safe_with_prev_pts]
            tempPts = tempPts[is_safe_with_prev_pts, :]

        if is_safe_to_continue > 0:
            # check with previously generated points
            is_safe_with_prev_pts = (
                KDTree(pts).query_radius(tempPts, r=spacing, count_only=True)==0
            )
            is_safe_to_continue = np.sum(is_safe_with_prev_pts)
            rejected_grids = p[~is_safe_with_prev_pts]
            scoreGrid[rejected_grids] = scoreGrid[rejected_grids] * discount_factor

            p = p[is_safe_with_prev_pts]
            tempPts = tempPts[is_safe_with_prev_pts, :]

        if is_safe_to_continue > 0:
            # find colliding pairs and leave only one of the pairs
            ind = KDTree(tempPts).query_radius(tempPts, r=spacing)
            is_eligible = np.frompyfunc(lambda i: (ind[i]<i).sum()==0, 1, 1)(np.arange(ind.size)).astype(bool)

            accepted_pts = tempPts[is_eligible, :]

            accepted_grids = p[is_eligible]
            rejected_grids = p[~is_eligible]
            remaining_grids = np.setdiff1d(range(sGrid.shape[0]), accepted_grids)

            sGrid = sGrid[remaining_grids, :]
            # scoreGrid[rejected_grids] = scoreGrid[rejected_grids] * discount_factor
            scoreGrid = scoreGrid[remaining_grids]

            # Update quantities for next iterations
            nEmptyGrid = sGrid.shape[0]
            if n_pts_created == 0:
                pts = accepted_pts
            else:
                pts = np.vstack((pts, accepted_pts))
            n_pts_newly_created = accepted_pts.shape[0]
            n_pts_created = pts.shape[0]

            if showIter:
                pbar.update(n_pts_newly_created)

        iter += 1

    if pts.shape[0] > nPts:
        p = np.arange(pts.shape[0])
        p = np.random.choice(p, nPts, replace=False)
        pts = pts[p, :]

    if showIter:
        pbar.close()
        print(
            "Iteration: {}, (final)Points Created: {}, is_grid_empty:{} ({}%)".format(
                iter, pts.shape[0], nEmptyGrid, nEmptyGrid / nEmptyGrid0 * 100
            )
        )
    return pts


Bridson_sampling_2d = partial(Bridson_sampling_1, make_2d_grid)

Bridson_sampling_3d = partial(Bridson_sampling_1, make_3d_grid)

