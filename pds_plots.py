import matplotlib.pyplot as plt

def plot_mf_1(mf_points, box):
    Horizontal_range, Transverse_range = box
    _, ax = plt.subplots(figsize=(15, 7))
    ax.scatter(mf_points[:, 0], mf_points[:, 1], 50, 'k')
    ax.set(xlim=[0, Horizontal_range],
        ylim=[0, Transverse_range])

    plt.axis('off')
    plt.tight_layout()
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


