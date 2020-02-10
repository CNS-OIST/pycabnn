import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm.autonotebook import tqdm

filename = "/Volumes/Sage2/shhong/Dropbox/Data/GINIX_data/cells_GINIX_GL.dat"
data = np.loadtxt(filename)

conversion_factor = 2 * 0.178
data = data * conversion_factor

xyz = data[:, :3]
nn = NearestNeighbors()
nn.fit(xyz)

dists, nnids = nn.kneighbors(xyz, n_neighbors=2, return_distance=True)
nnids = nnids[:, 1]
dists = dists[:, 1]

_, ax = plt.subplots(figsize=(8.9 / 2.54 / 1.75, 8.9 / 2.54 * 5 / 8 / 1.5))
ax.hist(dists, 450, density=True)
# _ = plt.hist(dists_u, 500)
ax.set(xlim=[2.5, 10], xlabel="", ylabel="", xticks=np.arange(4,12,2))
plt.tight_layout()
plt.savefig("nn_dist_hist_human.png", dpi=600, transparent=True)


def limit_to_box(x, box):
    mf = x.copy()
    for i, t in enumerate(box):
        mf = mf[mf[:, i] >= t[0], :]
        mf = mf[mf[:, i] <= t[1], :]
    return mf


gry = limit_to_box(xyz, [[30, 270], [230, 270], [30, 320]])
grx = xyz

nn = NearestNeighbors(n_jobs=-1)
nn.fit(grx)

mcounts = []
sdcounts = []
dists = np.linspace(0, 30, 240)
for r in tqdm(dists):
    count = (
        np.frompyfunc(lambda x: x.size, 1, 1)(
            nn.radius_neighbors(gry, radius=r, return_distance=False)
        ).astype(float)
        - 1
    )
    mcounts.append(count.mean())
    sdcounts.append(count.std() / np.sqrt(count.size))

cc2 = np.gradient(mcounts)/(dists**2)
cc2_0 = cc2[-1]
cc2 = cc2/cc2_0

mcounts = np.array(mcounts)
sdcounts = np.array(sdcounts)

cc2_u = np.gradient(mcounts + 150*sdcounts)/(dists**2+0.001)/cc2_0
cc2_d = np.gradient(mcounts - 150*sdcounts)/(dists**2+0.001)/cc2_0


_, ax = plt.subplots(figsize=(8.9/2.54/1.75, 8.9/2.54*5/8/1.5))
ax.plot(dists, cc2)
ax.set(
    ylim = [0, 3],
    xlim = [0, 30],
    xlabel='',
    ylabel=''
)
plt.tight_layout()
plt.savefig('cc2_grc_human.png', dpi=600, transparent=True)
