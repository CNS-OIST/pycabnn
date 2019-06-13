import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import norm
from IPython import embed
import time

dlat2 = np.array([1, 0, 0, 1, 1, 1]).reshape((-1, 2)).astype('double')

dlat3 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1]).reshape((-1, 3)).astype('double')


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
        if self.points.shape[0] >= points.shape[0]:
            nnsearch = KDTree(self.points).query_radius(points, r=self.r, count_only=True)

            return nnsearch == 0
        else:
            inds = KDTree(points).query_radius(points, r=self.r)
            inds = np.unique(np.hstack(inds))

            embed()
            return ~np.isin(range(points.shape[0]), inds)


    def test_cells(self, cell_corners, dgrid):
        nn2 = KDTree(cell_corners)
        inds = nn2.query_radius(self.points, r=self.r)
        is_covering1 = select_non_empty(inds)

        n_test = np.sum(is_covering1)
        selected_points = self.points[is_covering1, :]
        inds = inds[is_covering1]

        dlat = self.dlat.ravel() * dgrid
        t2 = time.time()
        ftest = lambda i: cells_each_point_covers(
            cell_corners[inds[i], :] - selected_points[i, :],
            inds[i],
            dlat,
            self.dim,
            self.r,
        )

        cells_covered = np.frompyfunc(ftest, 1, 1)(range(n_test))
        cells_covered = np.unique(np.hstack(cells_covered).astype(int))
        t3 = time.time()
        print(t3 - t2)

        return np.isin(range(cell_corners.shape[0]), cells_covered, invert=True)

