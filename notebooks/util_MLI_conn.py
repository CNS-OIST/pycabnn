import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from tqdm.autonotebook import trange


def compute_soma_dist(som):
    """computes the distance between each soma and the closest soma

    Args:
        som: soma coordinates

    Returns:
        Dictionary: soma-to-soma distance in xy and z directions
    """
    temp1 = pdist(som[:, [1, 2]])
    temp2 = pdist(som[:, 0][:, np.newaxis])
    som_dist = pd.DataFrame(
        np.zeros((temp1.shape[0], 2), dtype=int), columns=["i", "j"]
    )
    ifirst = 0
    ilast = som.shape[0] - 1
    for i in trange(som.shape[0] - 1):
        som_dist.iloc[ifirst:ilast, 0] = i
        som_dist.iloc[ifirst:ilast, 1] = np.arange(i + 1, som.shape[0], dtype=int)
        ifirst = ilast
        ilast = ifirst + som.shape[0] - (i + 2)

    som_dist["dist_xy"] = temp1
    som_dist["dist_z"] = temp2
    return som_dist


## Fitted distribution for the xy distance
def f_xy(x, mu, a, N):
    return N * np.exp(-a * (x - mu) * (x - mu)) * x


def pprior_xy(x):
    """Computes the p_data(x) for given xy distance x and gap junction"""

    return f_xy(x, -110, 1.4e-5, 1.3177)


def pdata_gap_xy(x):
    """Computes the p_data(x) for given xy distance x and gap junction"""
    return f_xy(x, 20.8, 2.6e-4, 0.53)


def p_gap_xy(x):
    return pdata_gap_xy(x) / pprior_xy(x)


def pdata_syn_xy(x):
    """Computes the p_data(x) for given xy distance x and synnaptic connection"""
    return f_xy(x, 33, 1.65e-4, 0.2)


## Fitted distribution for the z distance
def f_z(x, mu, a, N):
    return N * np.exp(-a * (x - mu) * (x - mu))


pprior_z = 1


def pdata_gap_z(x):
    """Computes the p_data(x) for given z distance x and gap junction"""
    return f_z(x, 5.8, 6.17e-3, 0.554)


def p_gap_z(x):
    return pdata_gap_z(x) / pprior_z


def pdata_syn_z(x):
    """Computes the p_data(x) for given z distance x and synnaptic connection"""
    return f_z(x, 13, 2.642e-3, .2337)


def p_syn_z(x):
    return pdata_syn_z(x) / pprior_z

