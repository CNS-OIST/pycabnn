import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_circles(mf_points, box, r, color, ax=None):
    Horizontal_range, Transverse_range = box

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    for p in mf_points:
        ax.add_artist(Circle((p[0], p[1]), radius=r, color=color))
    # ax.scatter(mf_points[:, 0], mf_points[:, 1], 50, 'k')
    ax.set(xlim=[0, Horizontal_range],
           ylim=[0, Transverse_range])

    plt.axis('off')
    plt.tight_layout()

    return ax


def plot_mf_1(mf_points, box, r):
    plot_circles(mf_points, box, r, color='k')
    plt.savefig('mf.png', dpi=300)
    # plt.show()


def plot_mf_2(mf_points, box):
    from scipy.spatial import Voronoi, voronoi_plot_2d

    Horizontal_range, Transverse_range = box

    vor = Voronoi(mf_points)
    fig = voronoi_plot_2d(vor, linewidth=0.01, point_size=8, show_vertices=False)
    plt.ylim([0, Transverse_range/4*1.75])
    plt.xlim([0, Horizontal_range/4*1.75*0.75])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('mf_vor.png', dpi=300)


def plot_slice(points, box, slice, r, color='k', ax=None):
    z = points[:, 2]
    indc = np.logical_and(z>slice[0], z<slice[1])
    spoints = points[indc, :]
    return plot_circles(spoints[:,:2], box[:2], r, color=color, ax=ax)


def plot_goc(points, box, slice, r):
    _ = plot_slice(points, box, slice, r)
    plt.savefig('goc.png', dpi=300)
    # plt.show()


def plot_glo(points, box, slice, r):
    _ = plot_slice(points, box, slice, r)
    plt.savefig('glo.png', dpi=300)
    # plt.show()


def plot_goc_glo(points_r1, points_r2, box, slice):
    points, r = points_r1
    ax = plot_slice(points, box, slice, r, color='g')

    points, r = points_r2
    _ = plot_slice(points, box, slice, r, color='c', ax=ax)
    plt.savefig('goc+glo.png', dpi=300)
    # plt.show()

def plot_all_pop(points_r1, points_r2, points_r3, box, slice):
    points, r = points_r1
    ax = plot_slice(points, box, slice, r, color='g')

    points, r = points_r2
    ax = plot_slice(points, box, slice, r, color='c', ax=ax)

    points, r = points_r3
    _ = plot_slice(points, box, slice, r, color='k', ax=ax)

    plt.savefig('goc+glo+grc.png', dpi=300)
    # plt.show()
