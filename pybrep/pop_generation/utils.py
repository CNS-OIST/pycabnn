import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import norm

dlat2 = np.array([1,0,
                  0,1,
                  1,1])

dlat3 = np.array([1,0,0,
                  0,1,0,
                  0,0,1,
                  1,1,0,
                  1,0,1,
                  0,1,1,
                  1,1,1])


def select_non_empty(ind):
    return np.frompyfunc(lambda x: x.size>0, 1, 1)(ind).astype(bool)


def cells_each_point_covers(vertices, point, ind, r):
    return [i for i in ind if np.all(norm(vertices[i, :, :] - point) <= r)]


class PointCloud(object):
    def __init__(self, points, r):
        super().__init__()
        self.points = points
        self.nn = KDTree(self.points)
        self.r = r

    def test_points(self, points):
        nnsearch = self.nn.query_radius(points, r=self.r, count_only=True)
        return (nnsearch == 0)

    def test_cell(self, cell_corners, dgrid):
        dim = cell_corners.shape[1]
        n_cells = cell_corners.shape[0]
        dlat = eval("dlat{}".format(dim))

        nn2 = KDTree(cell_corners)
        ind = nn2.query_radius(self.points, r=self.r)
        is_covering1 = select_non_empty(ind)

        n_test = np.sum(is_covering1)
        selected_points = self.points[is_covering1, :]
        inds = inds[is_covering1]
        vertices = (np.tile(cell_corners, (1, 2**dim-1)) + dlat*dgrid).reshape((n_cells, -1, dim))

        ftest = lambda i: cells_each_point_covers(vertices, selected_points[i, :], inds[i], self.r)
        # np.frompyfunc(ftest)(range(n_test))

