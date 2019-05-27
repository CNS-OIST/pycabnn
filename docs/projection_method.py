# %%
%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pf(ax):
    rand = np.random.rand
    xs = rand(2)*10 + np.array([600, 1200])
    ys = np.tile(rand(1)*100 + 300, 2)
    zs = np.tile(rand(1)*50 + 300, 2)
    ax.plot(xs, ys, zs, 'k')

# %%
def ball(ax, somc, r=15):

    import scipy.ndimage

    u = np.linspace(0, 2 * np.pi, 13)
    v = np.linspace(0, np.pi, 7)

    x = r * np.outer(np.cos(u), np.sin(v)) + somc[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + somc[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + somc[2]

    # use scipy to interpolate
    xdata = scipy.ndimage.zoom(x, 3)
    ydata = scipy.ndimage.zoom(y, 3)
    zdata = scipy.ndimage.zoom(z, 3)

    ax.plot_surface(xdata, ydata, zdata, rstride=3, cstride=3, color="k", shade=0)


def dends(ax, somc):
    ax.plot(
        np.tile(somc[0], 2),
        np.tile(somc[1], 2),
        np.array([somc[2], somc[2] + 50]),
        "k",
        linewidth=6,
    )

    ax.plot(
        np.array([somc[0], somc[0]+5]),
        np.array([somc[1], somc[1]-50]),
        np.array([somc[2]+50, somc[2] + 50 + 75]),
        "k",
        linewidth=3,
    )

    ax.plot(
        np.array([somc[0]+5, somc[0]+10]),
        np.array([somc[1]-50, somc[1]-50+30]),
        np.array([somc[2]+50+75, somc[2]+100+75+125]),
        "k",
        linewidth=2,
    )

    ax.plot(
        np.array([somc[0]+5, somc[0]]),
        np.array([somc[1]-50, somc[1]-50-30]),
        np.array([somc[2]+50+75, somc[2]+100+75+125]),
        "k",
        linewidth=2,
    )
    
    ax.plot(
        np.array([somc[0], somc[0]+5]),
        np.array([somc[1], somc[1]+50]),
        np.array([somc[2]+50, somc[2]+50+75]),
        "k",
        linewidth=3,
    )

    ax.plot(
        np.array([somc[0]+5, somc[0]]),
        np.array([somc[1]+50, somc[1]+50+30]),
        np.array([somc[2]+50+75, somc[2]+100+75+125]),
        "k",
        linewidth=2,
    )

    ax.plot(
        np.array([somc[0]+5, somc[0]+10]),
        np.array([somc[1]+50, somc[1]+50-30]),
        np.array([somc[2]+50+75, somc[2]+100+75+125]),
        "k",
        linewidth=2,
    )





def gcell(ax, somc=[750, 350, 100]):
    ball(ax, somc)
    dends(ax, somc)


# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
pf(ax)
pf(ax)
pf(ax)
gcell(ax)
ax.set(xlim=[600, 900], ylim=[200, 500], zlim=[100, 400])

# %%
plt.close('all')

# %%


# %%
