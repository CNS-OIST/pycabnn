"""Bridson sampling

References: Robert Bridson, Fast Poisson Disk Sampling in Arbitrary Dimensions , SIGGRAPH, 2007

Written by Sanghun Jee and Sungho Hong
Supervised by Erik De Schutter
Computational Neuroscience Unit,
Okinawa Institute of Science and Technology

March, 2020
"""

import numpy as np

from functools import partial
from tqdm.autonotebook import tqdm
from sklearn.neighbors import KDTree, NearestNeighbors

from .utils import PointCloud


def make_2d_grid(sizeI, cellsize):
    return np.mgrid[0 : sizeI[0] : cellsize, 0 : sizeI[1] : cellsize]


def make_3d_grid(sizeI, cellsize):
    return np.mgrid[
        0 : sizeI[0] : cellsize, 0 : sizeI[1] : cellsize, 0 : sizeI[2] : cellsize
    ]


def set_nDarts(nPts, n_pts_created, n_pts_newly_created, nEmptyGrid, dartFactor=4):
    n_to_create = nPts - n_pts_created
    # if n_pts_newly_created==0:
    #
    # else:
    #     ndarts = n_pts_newly_created*2
    ndarts = np.min([nPts / dartFactor, nEmptyGrid / dartFactor])
    # ndarts = np.max([n_to_create, nPts / dartFactor])
    return int(np.round(ndarts))


def bridson_sampling(
    fgrid,
    sizeI,
    spacing,
    nPts,
    showIter,
    ftests=[],
    discount_factor=0.5,
    stopping_criterion="density",
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

    # Initialize Parameters
    if nPts == 0:
        nPts = nEmptyGrid
    n_pts_created = 0
    n_pts_newly_created = 0
    pts = np.empty(shape=(1, ndim))
    iter = 0

    # Start Iterative process

    pcl = PointCloud(pts, spacing)
    nn2 = NearestNeighbors(radius=spacing)

    if ftests != []:
        print("Testing coverage of cells...")
        is_cell_uncovered = np.ones(sGrid.shape[0], dtype=bool)
        for ftest in ftests:
            is_cell_uncovered = ftest.test_cells(sGrid, dgrid)

            sGrid = sGrid[is_cell_uncovered, :]
            scoreGrid = scoreGrid[is_cell_uncovered]
            nEmptyGrid = np.sum(sGrid.shape[0])
            print("Uncovered cells: {}%\n".format(nEmptyGrid / nEmptyGrid0 * 100))

    if showIter:
        pbar = tqdm(total=nPts)
        pbar_grid = tqdm(total=nEmptyGrid0)
        pbar_grid.update(nEmptyGrid0 - nEmptyGrid)

    if stopping_criterion != 'density':
        raise RuntimeError('Bridson sampling  allows only the density-based stopping criterion')

    while n_pts_created < nPts and nEmptyGrid > 0:
        # Thrown darts in eligible grids
        # availGrid, = np.where(is_grid_empty)
        # score_availGrid = scoreGrid[availGrid]
        scoreGrid = scoreGrid / scoreGrid.sum()

        ndarts = set_nDarts(nPts, n_pts_created, n_pts_newly_created, nEmptyGrid)
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
                is_safe_with_prev_pts = is_safe_with_prev_pts * ftest.test_points(
                    tempPts
                )

            is_safe_to_continue = np.sum(is_safe_with_prev_pts)
            rejected_grids = p[~is_safe_with_prev_pts]
            scoreGrid[rejected_grids] = scoreGrid[rejected_grids] * discount_factor

            p = p[is_safe_with_prev_pts]
            tempPts = tempPts[is_safe_with_prev_pts, :]

        if is_safe_to_continue > 0 and n_pts_created > 0:
            # check with previously generated points
            # is_safe_with_prev_pts = np.frompyfunc(lambda x: x.size == 0, 1, 1)(
            #     nn1.radius_neighbors(tempPts, return_distance=False)
            # ).astype(bool)
            is_safe_with_prev_pts = pcl.test_points(tempPts)
            is_safe_to_continue = np.sum(is_safe_with_prev_pts)
            rejected_grids = p[~is_safe_with_prev_pts]
            scoreGrid[rejected_grids] = scoreGrid[rejected_grids] * discount_factor

            p = p[is_safe_with_prev_pts]
            tempPts = tempPts[is_safe_with_prev_pts, :]

        if is_safe_to_continue > 0:
            # find colliding pairs and leave only one of the pairs
            nn2.fit(tempPts)
            ind = nn2.radius_neighbors(tempPts, return_distance=False)

            # ind = KDTree(tempPts).query_radius(tempPts, r=spacing)
            is_eligible = np.frompyfunc(lambda i: (ind[i] < i).sum() == 0, 1, 1)(
                np.arange(ind.size)
            ).astype(bool)

            accepted_pts = tempPts[is_eligible, :]

            accepted_grids = p[is_eligible]
            rejected_grids = p[~is_eligible]
            remaining_grids = np.setdiff1d(range(sGrid.shape[0]), accepted_grids)

            sGrid = sGrid[remaining_grids, :]
            # scoreGrid[rejected_grids] = scoreGrid[rejected_grids] * discount_factor
            scoreGrid = scoreGrid[remaining_grids]

            n_pts_newly_created = accepted_pts.shape[0]
            if n_pts_newly_created > 0:
                # Update quantities for next iterations
                nEmptyGrid = sGrid.shape[0]
                if n_pts_created == 0:
                    pcl.update_points(accepted_pts)
                else:
                    pcl.append_points(accepted_pts)

                n_pts_created = pcl.points.shape[0]

            if showIter:
                pbar.update(n_pts_newly_created)
                pbar_grid.update(n_pts_newly_created)

        iter += 1

    pts = pcl.points
    if pts.shape[0] > nPts:
        p = np.arange(pts.shape[0])
        p = np.random.choice(p, nPts, replace=False)
        pts = pts[p, :]

    if showIter:
        pbar.close()
        pbar_grid.close()
        print(
            "\nIteration: {}, (final)Points Created: {}, is_grid_empty:{} ({}%)".format(
                iter, pts.shape[0], nEmptyGrid, nEmptyGrid / nEmptyGrid0 * 100
            )
        )
    return pts


Bridson_sampling_2d = partial(bridson_sampling, make_2d_grid)

Bridson_sampling_3d = partial(bridson_sampling, make_3d_grid)
