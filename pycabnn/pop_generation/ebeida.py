"""Maximal Poisson disk sampling

Reference: Ebeida, Mohamed S., et al. "A simple algorithm for maximal Poisson‐disk sampling in high dimensions." Computer Graphics Forum. Vol. 31. No. 2pt4. Oxford, UK: Blackwell Publishing Ltd, 2012.

Written by Sanghun Jee and Sungho Hong
Supervised by Erik De Schutter
Computational Neuroscience Unit,
Okinawa Institute of Science and Technology

March, 2020
"""


import numpy as np
from functools import partial
from sklearn.neighbors import KDTree, NearestNeighbors
from tqdm.autonotebook import tqdm

from .utils import PointCloud

dlat2 = np.array([0, 0, 1, 0, 0, 1, 1, 1]).reshape((-1, 2)).astype("double")

dlat3 = (
    np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1])
    .reshape((-1, 3))
    .astype("double")
)


def make_2d_grid(sizeI, cellsize):
    return np.mgrid[0 : sizeI[0] : cellsize, 0 : sizeI[1] : cellsize]


def make_3d_grid(sizeI, cellsize):
    return np.mgrid[
        0 : sizeI[0] : cellsize, 0 : sizeI[1] : cellsize, 0 : sizeI[2] : cellsize
    ]


def set_nDarts(nPts, n_pts_created, n_pts_newly_created, n_empty_cells, dartFactor=5):
    n_to_create = nPts - n_pts_created
    ### Alternative choice for ndarts
    # if n_pts_newly_created==0:
    #
    # else:
    #     ndarts = n_pts_newly_created*2
    # ndarts = np.min([nPts / dartFactor, n_empty_cells / dartFactor])
    # ndarts = np.max([n_to_create, nPts / dartFactor])

    ndarts = n_empty_cells / dartFactor

    return int(np.round(ndarts))


def ebeida_sampling(
    sizeI,
    spacing,
    nPts,
    showIter,
    ftests=[],
    discount_factor=0.5,
    stop_method="density",
):

    # Setting properties of iteration
    ndim = len(sizeI)
    cellsize = spacing / np.sqrt(ndim)
    fgrid = eval("make_{}d_grid".format(ndim))
    dlat = eval("dlat{}".format(ndim)).ravel()

    gen_history = np.zeros(10)

    # Make grid size such that there is just one pt in each grid
    dcell = spacing / np.sqrt(ndim)

    # Make a grid and convert it into a nxD array
    s_cell = fgrid(sizeI, cellsize)
    s_cell = np.array([s_cell[i][:].flatten() for i in range(ndim)]).T
    grid = np.arange(s_cell.shape[0]).astype(int)

    # Thrown in a particular grid
    n_empty_cells = n_empty_cells0 = s_cell.shape[0]

    # Initialize Parameters
    if nPts == 0:
        nPts = n_empty_cells
    n_pts_created = 0
    n_pts_newly_created = 0
    pts = np.empty(shape=(1, ndim))
    # iter = 0

    # Start Iterative process
    pcl = PointCloud(pts, spacing)
    nn2 = NearestNeighbors(radius=spacing, algorithm="kd_tree", leaf_size=50)

    if ftests != []:
        for ftest in ftests:
            is_cell_uncovered = ftest.test_cells(s_cell, dcell)

            s_cell = s_cell[is_cell_uncovered, :]
            grid = grid[is_cell_uncovered]
            n_empty_cells = np.sum(s_cell.shape[0])

    if showIter:
        pbar = tqdm(total=nPts)

    if stop_method == 'density':
        fcontinue_criterion = lambda n_pts, n_empty: n_pts < nPts and n_empty > 0
    elif stop_method == 'maximal':
        fcontinue_criterion = lambda n_pts, n_empty: n_empty > 0
    else:
        raise RuntimeError(f'Unknown stopping criterion: {stop_method}')

    while fcontinue_criterion(n_pts_created, n_empty_cells):
        # Thrown darts in eligible grids

        ndarts = set_nDarts(nPts, n_pts_created, n_pts_newly_created, n_empty_cells)
        if ndarts != s_cell.shape[0]:
            p = np.random.choice(range(s_cell.shape[0]), ndarts, replace=False)
        else:
            p = range(s_cell.shape[0])

        is_safe_to_continue = 1

        tempPts = s_cell[p, :] + dcell * np.random.rand(len(p), ndim)
        temp_grids = grid[p]

        if ftests != []:
            is_safe_with_prev_pts = np.ones(len(p), dtype=bool)
            for ftest in ftests:
                is_safe_with_prev_pts = is_safe_with_prev_pts * ftest.test_points(
                    tempPts
                )

            p = p[is_safe_with_prev_pts]
            tempPts = tempPts[is_safe_with_prev_pts, :]
            temp_grids = temp_grids[is_safe_with_prev_pts]

        is_safe_to_continue = p.size  # tempPts.shape[0]

        if is_safe_to_continue > 0 and n_pts_created > 0:
            is_safe_with_prev_pts = pcl.test_points(tempPts)
            is_safe_to_continue = np.sum(is_safe_with_prev_pts)

            p = p[is_safe_with_prev_pts]
            tempPts = tempPts[is_safe_with_prev_pts, :]
            temp_grids = temp_grids[is_safe_with_prev_pts]

        _, ind = np.unique(temp_grids, return_index=True)
        is_unlocked = np.isin(range(p.size), ind)
        is_safe_to_continue = np.sum(is_unlocked)
        p = p[is_unlocked]
        tempPts = tempPts[is_unlocked, :]
        temp_grids = temp_grids[is_unlocked]

        if is_safe_to_continue > 0:
            # find colliding pairs and leave only one of the pairs
            nn2.fit(tempPts)
            ind = nn2.radius_neighbors(tempPts, return_distance=False)

            is_eligible = np.frompyfunc(lambda i: (ind[i] < i).sum() == 0, 1, 1)(
                np.arange(ind.size)
            ).astype(bool)

            n_pts_newly_created = np.sum(is_eligible)
            rejection_rate = 1 - n_pts_newly_created / ndarts

            gen_history = np.roll(gen_history, 1)
            gen_history[0] = n_pts_newly_created

            if n_pts_newly_created > 0:
                accepted_pts = tempPts[is_eligible, :]
                accepted_grids = temp_grids[is_eligible]

                is_grid_unmarked = ~np.isin(grid, accepted_grids)
                s_cell = s_cell[is_grid_unmarked, :]
                grid = grid[is_grid_unmarked]

                # Update quantities for next iterations
                n_empty_cells = s_cell.shape[0]
                if n_pts_created == 0:
                    pcl.update_points(accepted_pts)
                else:
                    pcl.append_points(accepted_pts)

                n_pts_created = pcl.points.shape[0]

                if showIter:
                    pbar.update(n_pts_newly_created)
                    # print('n_pts_created = ', n_pts_created, '/', nPts)

        is_safe_to_continue = s_cell.shape[0]

        # if is_safe_to_continue and n_pts_newly_created/nPts<0.0006:
        if is_safe_to_continue and n_pts_newly_created < 0.0006 * nPts:
            print("Splitting grids...")
            dcell = dcell / 2
            s_cell = (np.tile(s_cell, (1, 2 ** ndim)) + dlat * dcell).reshape(
                (-1, ndim)
            )
            grid = np.repeat(grid, 2 ** ndim)
            n_empty_cells0 = np.sum(s_cell.shape[0])
            assert grid.size == n_empty_cells0

            if ftests != []:
                for ftest in ftests:
                    is_cell_uncovered = ftest.test_cells(s_cell, dcell)

                    s_cell = s_cell[is_cell_uncovered, :]
                    grid = grid[is_cell_uncovered]
                    n_empty_cells = np.sum(s_cell.shape[0])

            n_empty_cells0 = np.sum(s_cell.shape[0])
            is_cell_uncovered = pcl.test_cells(s_cell, dcell)

            s_cell = s_cell[is_cell_uncovered, :]
            grid = grid[is_cell_uncovered]
            n_empty_cells = n_empty_cells0 = np.sum(s_cell.shape[0])

    pts = pcl.points
    if stop_method == 'density' and pts.shape[0] > nPts:
        p = np.arange(pts.shape[0])
        p = np.random.choice(p, nPts, replace=False)
        pts = pts[p, :]

    if showIter:
        pbar.close()

    return pts


ebeida_sampling_2d = partial(ebeida_sampling, make_2d_grid)

ebeida_sampling_3d = partial(ebeida_sampling, make_3d_grid)
