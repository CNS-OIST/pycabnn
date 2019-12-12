#!/usr/bin/env python
"""run_pybrep.py

Jobs = AAtoGoC, PFtoGoC, GoCtoGoC, GoCtoGoCgap

Usage:
  run_pybrep.py (-i PATH) (-o PATH) (-p PATH) [--parallel] (all | <jobs>...)
  run_pybrep.py (-h | --help)
  run_pybrep.py --version

Options:
  -h --help                            Show this screen.
  --version                            Show version.
  -i PATH, --input_path=<input_path>   Input path.
  -o PATH, --output_path=<output_path> Output path.
  -p PATH, --param_path=<param_dir>    Params path.
  --parallel                           Run things in parallel (via ipyparallel)

"""

from pathlib import Path
import pybrep as brp
import time
from docopt import docopt
from neuron import h
import numpy as np


def load_input_data(args):
    data = {}
    for arg in args:
        if "path" in arg:
            data[arg.replace("--", "")] = Path(args[arg]).resolve()

    data["config_hoc"] = data["param_path"] / "Parameters.hoc"

    t1 = time.time()
    print("Starting parallel process...")

    # Read in the config file(or the pseudo-hoc, see pseudo-hoc class to find out how it is generated)
    try:
        from neuron import h

        config_hoc = str(data["config_hoc"])
        h.xopen(config_hoc)
        print("Trying to read in hoc config object from ", config_hoc)
    except ModuleNotFoundError:
        # TODO: have to find a good way to deal with this
        config_pseudo_hoc = Path.cwd() / "pseudo_hoc.pkl"
        h = brp.Pseudo_hoc(config_pseudo_hoc)
        print("Trying to read in pseudo-hoc config object from ", config_pseudo_hoc)
    finally:
        # Just pick a random variable and check whether it is read
        assert hasattr(
            h, "GoC_Atheta_min"
        ), "There might have been a problem reading in the parameters!"
        print("Succesfully read in config file!")

    t2 = time.time()
    print("Import finished:", t2 - t1)
    data["t"] = t2
    return data


def load_and_make_population(data, pops):

    output_path = data["output_path"]

    def make_glo(data):
        """sets up the Glomerulus population"""
        from pybrep.cell_population import Cell_pop
        glo_in = data["input_path"] / "GLcoordinates.dat"

        glo = Cell_pop(h)
        glo.load_somata(glo_in)
        glo.save_somata(output_path, "GLcoordinates.sorted.dat")

        t0 = data["t"]
        t1 = time.time()
        print("GL processing:", t1 - t0)
        return (glo, t1)


    def make_goc(data):
        """sets up the Golgi population, render dendrites."""
        gol_in = data["input_path"] / "GoCcoordinates.dat"
        gg = brp.create_population("Golgi", h)
        gg.load_somata(gol_in)
        # # gg.gen_random_cell_loc(1995, 1500, 700, 200)
        gg.add_dendrites()
        gg.save_dend_coords(output_path)
        gg.add_axon()
        gg.save_somata(output_path, "GoCcoordinates.sorted.dat")

        t2 = data["t"]
        t3 = time.time()
        print("Golgi cell processing:", t3 - t2)
        return (gg, t3)


    def make_grc(data):
        """sets up Granule population including aa and pf."""
        gran_in = data["input_path"] / "GCcoordinates.dat"
        gp = brp.create_population("Granule", h)
        gp.load_somata(gran_in)
        gp.add_aa_endpoints_fixed()
        gp.add_pf_endpoints()
        gp.save_gct_points(output_path)
        gp.save_somata(output_path, "GCcoordinates.sorted.dat")

        t3 = data["t"]
        t4 = time.time()
        print("Granule cell processing:", t4 - t3)
        return (gp, t4)

    print(" ")

    data["pops"] = {}
    for c in pops:
        data["pops"][c], t = eval('make_'+c)(data)
        data["t"] = t

    return data


def run_AAtoGoC(data):
    # you might want to change the radii
    c_rad_aa = h.AAtoGoCzone / 1.73
    print("R for AA: {}".format(c_rad_aa))

    gp, gg = data["pops"]["grc"], data["pops"]["goc"]
    cc = brp.Connect_2D(gp.qpts_aa, gg.qpts, c_rad_aa)
    # TODO: adapt the following from command line options
    cc.connections_parallel(deparallelize=False, nblocks=120, debug=False)
    cc.save_result(data["output_path"] / "AAtoGoC")

    t5 = time.time()
    t4 = data["t"]
    print("AA: Found and saved after", t5 - t4)
    print(" ")
    data["t"] = t5
    return data


def run_PFtoGoC(data):
    c_rad_pf = h.PFtoGoCzone / 1.113
    print("R for PF: {}".format(c_rad_pf))

    gp, gg = data["pops"]["grc"], data["pops"]["goc"]
    cc = brp.Connect_2D(gp.qpts_pf, gg.qpts, c_rad_pf)
    cc.connections_parallel(deparallelize=False, nblocks=120, debug=False)
    cc.save_result(data["output_path"] / "PFtoGoC")

    t6 = time.time()
    t5 = data["t"]
    print("PF: Found and saved after", t6 - t5)
    print(" ")
    data["t"] = t6
    return data


def run_GoCtoGoC(data):
    import numpy as np
    from tqdm.autonotebook import tqdm
    from sklearn.neighbors import KDTree
    from pybrep.util import Query_point as qp

    output_path = data["output_path"]

    gg = data["pops"]["goc"]

    ## Old brute-force approach
    # for i in tqdm(range(gg.n_cell)):
    #     axon_coord1 = gg.axon[i]
    #     tree = KDTree(axon_coord1)
    #     for j in range(gg.n_cell):
    #         if i != j:
    #             ii, di = tree.query_radius(
    #                 np.expand_dims(gg.som[j], axis=0),
    #                 r=h.GoCtoGoCzone,
    #                 return_distance=True,
    #             )
    #             if ii[0].size > 0:
    #                 temp = di[0].argmin()
    #                 ii, di = ii[0][temp], di[0][temp]
    #                 axon_len = np.linalg.norm(axon_coord1[ii] - gg.som[i])
    #                 src.append(i)
    #                 tgt.append(j)
    #                 dist.append(axon_len + di)  # putative path length along the axon

    # Golgi cell axon points
    gax = qp(gg.axon)

    ii, di = KDTree(gax.coo).query_radius(
        gg.som, r=h.GoCtoGoCzone, return_distance=True
    )

    # Find all sources and exclude self-connections
    srcs = [np.unique(gax.idx[ix].ravel()) for ix in ii]
    srcs = [s[s!=n] for n, s in enumerate(srcs)]

    # Compute the soma-axon distance
    axdst = gax.coo-gg.som[gax.idx.ravel(),:]
    axdst = np.sqrt(np.sum(axdst**2, axis=1))

    src = []
    tgt = []
    dist = []

    # Compute soma-axon-soma distance and collect data
    for n, _ in enumerate(srcs):
        for s in srcs[n]:
            idx_axon = (gax.idx[ii[n]].ravel()==s)
            dst_ax_som = di[n][idx_axon]
            i_nearest, d_nearest = dst_ax_som.argmin(), dst_ax_som.min()
            i_nearest_axon = ii[n][idx_axon][i_nearest]
            d_axon = axdst[i_nearest_axon]
            dist.append(d_nearest + d_axon)
            src.append(s)
            tgt.append(n)

    np.savetxt(output_path / "GoCtoGoCsources.dat", src, fmt="%d")
    np.savetxt(output_path / "GoCtoGoCtargets.dat", tgt, fmt="%d")
    np.savetxt(output_path / "GoCtoGoCdistances.dat", dist)

    t2 = time.time()
    t1 = data["t"]
    print("GoCtoGoC: Found and saved after", t2 - t1)
    data["t"] = t2
    return data


def run_GoCtoGoCgap(data):
    import numpy as np
    from tqdm import tqdm
    from sklearn.neighbors import KDTree

    gg = data["pops"]["goc"]
    dist = []
    src = []
    tgt = []

    ## Old brute-force approach
    # for i in tqdm(range(gg.n_cell)):
    #     for j in range(gg.n_cell):
    #         if i != j:
    #             di = np.linalg.norm(gg.som[j] - gg.som[i])
    #             if di < h.GoCtoGoCgapzone:
    #                 src.append(i)
    #                 tgt.append(j)
    #                 dist.append(di)

    # Find all pairs within GoCtoGoCgapzone
    ii, di = KDTree(gg.som).query_radius(
            gg.som, r=h.GoCtoGoCgapzone, return_distance=True
    )

    srcs = ii
    srcs = [s[s!=n] for n, s in enumerate(ii)]
    dists = [di[n][s!=n] for n, s in enumerate(ii)]

    for n, _ in enumerate(srcs):
        for m, s in enumerate(srcs[n]):
            tgt.append(n)
            src.append(s)
            dist.append(dists[n][m])

    output_path = data["output_path"]
    np.savetxt(output_path / "GoCtoGoCgapsources.dat", src, fmt="%d")
    np.savetxt(output_path / "GoCtoGoCgaptargets.dat", tgt, fmt="%d")
    np.savetxt(output_path / "GoCtoGoCgapdistances.dat", dist)

    t2 = time.time()
    t1 = data["t"]
    print("GoCtoGoCgap: Found and saved after", t2 - t1)
    data["t"] = t2
    return data

def run_GlotoGrC(data):
    raise NotImplementedError()
    # from sklearn.neighbors import NearestNeighbors

    # glo, grc = data["pops"]["glo"], data["pops"]["grc"]

    # def squeezed_som_coord(x):
    #     y = x.som.copy()
    #     y[:,1] = y[:,1]/3
    #     return y

    # glo_som_squeezed = squeezed_som_coord(glo)
    # grc_som_squeezed = squeezed_som_coord(grc)

    # nn = NearestNeighbors()
    # nn.fit(glo_som_squeezed)


def main(args):
    data = load_input_data(args)
    # data = load_and_make_population(data, ["glo", "grc"])
    data = load_and_make_population(data, ["grc", "goc"])
    print(data)

    valid_job_list = ["AAtoGoC", "PFtoGoC", "GoCtoGoC", "GoCtoGoCgap"]

    if args["all"]:
        args["<jobs>"] = valid_job_list

    for j in args["<jobs>"]:
        if j not in valid_job_list:
            raise RuntimeError(
                "Job {} is not valid, not in {}".format(j, valid_job_list)
            )
        else:
            data = eval("run_" + j)(data)


if __name__ == "__main__":
    args = docopt(__doc__, version="0.7dev")
    main(args)
