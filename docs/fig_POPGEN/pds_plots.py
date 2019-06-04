import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm.autonotebook import tqdm

plt.style.use('dark_background')

def plot_circles(mf_points, box, r, color, ec=None, ax=None):
    xy = mf_points-np.array([25, 25])
    Horizontal_range, Transverse_range = box

    if ax is None:
        bboxr = (box[1][1]-box[1][0])/(box[0][1]-box[0][0])
        _ = plt.figure(figsize=(8.5, 8.5*bboxr))
        ax = plt.subplot(111)

    if type(r)==np.ndarray:
        for i, p in tqdm(enumerate(mf_points)):
            ax.add_artist(Circle((p[0], p[1]), radius=r[i], facecolor=color, edgecolor=ec))
    else:
        for p in tqdm(mf_points):
            ax.add_artist(Circle((p[0], p[1]), radius=r, facecolor=color, edgecolor=ec))
    # ax.scatter(mf_points[:, 0], mf_points[:, 1], 50, 'k')

    ax.set(xlim=Horizontal_range,
           ylim=Transverse_range,
           xlabel=None, ylabel=None, xticks=[], yticks=[])

    plt.axis('off')

    plt.subplots_adjust(
        left=0.0, right=1.0, top=1.0, bottom=0.0,
    )

    return ax


def plot_mf_1(mf_points, box, r, save=False):
    ax = plot_circles(mf_points, box, r, color='w')

    return ax


def plot_mf_2(mf_points, box, save=False):
    from scipy.spatial import Voronoi, voronoi_plot_2d

    Horizontal_range, Transverse_range = box

    vor = Voronoi(mf_points)
    fig = voronoi_plot_2d(vor, linewidth=0.01, point_size=8, show_vertices=False)
    plt.ylim([0, Transverse_range/4*1.75])
    plt.xlim([0, Horizontal_range/4*1.75*0.75])
    plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig('mf_vor.png', dpi=300)


def plot_slice(points, box, z_focal, r, color='w', ec=None, ax=None):
    z = points[:, 2]
    zdist = np.abs(z-z_focal)
    indc = (zdist < r)
    spoints = points[indc, :]
    r_focal = np.sqrt(r**2 - zdist[indc]**2)
    return plot_circles(spoints[:,:2], box[:2], r_focal, color=color, ec=ec, ax=ax)


def plot_goc(points, box, slice, r, ax=None):
    return plot_slice(points, box, slice, r, color='grey', ax=ax)
    # plt.show()


def plot_glo(points, box, slice, r, ax=None):
    return plot_slice(points, box, slice, r, color='y', ax=ax)
    # plt.show()

def plot_grc(points, box, slice, r, ax=None):
    return plot_slice(points, box, slice, r, color='b', ec='r', ax=ax)
    # plt.show()


def plot_goc_glo(points_r1, points_r2, box, slice):
    points, r = points_r1
    ax = plot_goc(points, box, slice, r)

    points, r = points_r2
    return plot_glo(points, box, slice, r, ax=ax)


def plot_all_pop(points_r1, points_r2, points_r3, box, slice):
    points, r = points_r1
    ax = plot_goc(points, box, slice, r)

    points, r = points_r2
    ax = plot_glo(points, box, slice, r, ax=ax)

    points, r = points_r3
    return plot_grc(points, box, slice, r, ax=ax)

    # plt.show()
