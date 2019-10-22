# References: Fast Poisson Disk Sampling in Arbitrary Dimensions
#             Robert Bridson, SIGGRAPH, 2007

import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors
from .utils import PointCloud
from tqdm.autonotebook import tqdm


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
    sizeI, spacing, nPts, ftests=[], discount_factor=0.5, show_progress=True
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
    fgrid = eval("make_{}d_grid".format(ndim))

    grid = fgrid(sizeI, cellsize)
    grid = np.array([grid[i][:].flatten() for i in range(ndim)]).T

    # Thrown in a particular grid
    n_empty_grid = n_empty_grid0 = grid.shape[0]
    score_grid = np.ones(grid.shape[0])

    # Initialize Parameters
    if nPts == 0:
        nPts = n_empty_grid
    n_pts_created = 0
    n_pts_newly_created = 0
    pts = np.empty(shape=(1, ndim))
    iter = 0

    # Start Iterative process

    pcl = PointCloud(pts, spacing)
    nn2 = NearestNeighbors(radius=spacing)

    if ftests != []:
        print("Testing coverage of cells...")
        is_cell_uncovered = np.ones(grid.shape[0], dtype=bool)
        for ftest in ftests:
            is_cell_uncovered = ftest.test_cells(grid, dgrid)

            grid = grid[is_cell_uncovered, :]
            score_grid = score_grid[is_cell_uncovered]
            n_empty_grid = np.sum(grid.shape[0])
            print("Uncovered cells: {}%\n".format(n_empty_grid / n_empty_grid0 * 100))

    if show_progress:
        pbar = tqdm(total=nPts)
        pbar_grid = tqdm(total=n_empty_grid0)
        pbar_grid.update(n_empty_grid0 - n_empty_grid)

    while n_pts_created < nPts and n_empty_grid > 0:
        score_grid = score_grid / score_grid.sum()

        ndarts = set_nDarts(nPts, n_pts_created, n_pts_newly_created, n_empty_grid)

        ndarts = int(np.round(n_empty_grid * 0.9))

        p = np.random.choice(range(grid.shape[0]), ndarts, replace=False, p=score_grid)
        temp_pts = grid[p, :] + dgrid * np.random.rand(len(p), ndim)

        is_safe_to_continue = 1

        if ftests != []:
            is_safe_with_prev_pts = np.ones(len(p), dtype=bool)
            for ftest in ftests:
                is_safe_with_prev_pts = is_safe_with_prev_pts * ftest.test_points(
                    temp_pts
                )

            is_safe_to_continue = np.sum(is_safe_with_prev_pts)
            rejected_grids = p[~is_safe_with_prev_pts]
            score_grid[rejected_grids] = score_grid[rejected_grids] * discount_factor

            p = p[is_safe_with_prev_pts]
            temp_pts = temp_pts[is_safe_with_prev_pts, :]

        accepted_grids, accepted_pts = test_new_points_with_existing(
            pcl,
            temp_pts,
            p,
            is_safe_to_continue,
            n_pts_created,
            nn2,
            score_grid=score_grid,
            discount_factor=discount_factor,
        )

        if len(accepted_grids) > 0:

            grid = np.delete(grid, accepted_grids, axis=0)
            score_grid = np.delete(score_grid, accepted_grids, axis=0)

            n_pts_newly_created = accepted_pts.shape[0]
            if n_pts_newly_created > 0:
                if n_pts_created == 0:
                    pcl.update_points(accepted_pts)
                else:
                    pcl.append_points(accepted_pts)

                n_pts_created = pcl.points.shape[0]
                n_empty_grid = grid.shape[0]

            if show_progress:
                pbar.update(n_pts_newly_created)
                pbar_grid.update(n_pts_newly_created)

        iter += 1

    pts = pcl.points
    if pts.shape[0] > nPts:
        p = np.arange(pts.shape[0])
        p = np.random.choice(p, nPts, replace=False)
        pts = pts[p, :]

    if show_progress:
        pbar.close()
        pbar_grid.close()
        print(
            "\nIteration: {}, (final)Points Created: {}, is_grid_empty:{} ({}%)".format(
                iter, pts.shape[0], n_empty_grid, n_empty_grid / n_empty_grid0 * 100
            )
        )
    return pts


def test_new_points_with_existing(
    pcl,
    temp_pts,
    p,
    is_safe_to_continue,
    n_pts_created,
    nn2,
    score_grid=[],
    discount_factor=1,
):

    accepted_grids = []
    accepted_pts = []

    # check with previously generated points
    if is_safe_to_continue > 0 and n_pts_created > 0:

        is_safe_with_prev_pts = pcl.test_points(temp_pts)
        is_safe_to_continue = np.sum(is_safe_with_prev_pts)

        rejected_grids = p[~is_safe_with_prev_pts]

        if len(score_grid) > 0:
            score_grid[rejected_grids] = score_grid[rejected_grids] * discount_factor

        p = p[is_safe_with_prev_pts]
        temp_pts = temp_pts[is_safe_with_prev_pts, :]

    # find colliding pairs and leave only one of the pairs
    if is_safe_to_continue > 0:
        nn2.fit(temp_pts)
        ind = nn2.radius_neighbors(temp_pts, return_distance=False)

        # select only the points which collides with itself or lower rank points
        is_eligible = np.frompyfunc(lambda i: (ind[i] < i).sum() == 0, 1, 1)(
            np.arange(ind.size)
        ).astype(bool)

        accepted_pts = temp_pts[is_eligible, :]
        accepted_grids = p[is_eligible]

    return accepted_grids, accepted_pts

