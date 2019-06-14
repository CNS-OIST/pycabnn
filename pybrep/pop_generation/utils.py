import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from IPython import embed

import time

dlat2 = np.array([1, 0, 0, 1, 1, 1]).reshape((-1, 2)).astype("double")

dlat3 = (
    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1])
    .reshape((-1, 3))
    .astype("double")
)


def select_non_empty(ind):
    return np.frompyfunc(lambda x: x.size > 0, 1, 1)(ind).astype(bool)


def cells_each_point_covers(dpoints, ind, dlat, dim, r):
    vertices = (np.tile(dpoints, (1, 2 ** dim - 1)) + dlat).reshape(
        (dpoints.shape[0], -1, dim)
    )
    return ind[(norm(vertices, axis=-1) <= r).all(axis=-1)]


class PointCloud(object):
    def __init__(self, points, r):
        super().__init__()
        self.points = points
        self.r = r
        self.dim = self.points.shape[1]
        self.dlat = eval("dlat{}".format(self.dim))

    def update_points(self, points):
        self.points = points

    def append_points(self, new_points):
        self.points = np.vstack((self.points, new_points))

    def test_points(self, points):
        nn1 = NearestNeighbors(radius=self.r, n_jobs=-1)
        if self.points.shape[0] >= points.shape[0]:
            nn1.fit(self.points)
            nnsearch = nn1.radius_neighbors(points, return_distance=False)
            )
            return ~select_non_empty(nnsearch)
        else:
            nn1.fit(points)
            inds = nn1.radius_neighbors(self.points, return_distance=False)
            inds = np.unique(np.hstack(inds))
            return ~np.isin(range(points.shape[0]), inds)

    def test_cells(self, cell_corners, dgrid, nn=None, return_nn=False):
        t0 = time.time()
        # nn2 = KDTree(cell_corners)

        if nn is None:
            nn2 = NearestNeighbors(n_jobs=-1)
            nn2.fit(cell_corners)
            t1 = time.time()
            print(t1 - t0)
        else:
            nn2 = nn
            t1 = time.time()

        # nn2 = KDTree(cell_corners)
        # inds = nn2.query_radius(self.points, r=self.r)
        inds = nn2.radius_neighbors(self.points, radius=self.r, return_distance=False)
        is_covering1 = select_non_empty(inds)

        n_test = np.sum(is_covering1)
        selected_points = self.points[is_covering1, :]
        inds = inds[is_covering1]

        dlat = self.dlat.ravel() * dgrid
        t2 = time.time()
        print(t2 - t1)

        ftest = lambda i: cells_each_point_covers(
            cell_corners[inds[i], :] - selected_points[i, :],
            inds[i],
            dlat,
            self.dim,
            self.r,
        )

        print("ntest: ", n_test)
        cells_covered = np.frompyfunc(ftest, 1, 1)(range(n_test))
        cells_covered = np.unique(np.hstack(cells_covered).astype(int))
        t3 = time.time()
        print(t3 - t2)

        if return_nn:
            return (np.isin(range(cell_corners.shape[0]), cells_covered, invert=True), nn2)
        else:
            return np.isin(range(cell_corners.shape[0]), cells_covered, invert=True)

