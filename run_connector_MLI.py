#!/usr/bin/env python
"""run_connector.py

Jobs = AAtoGoC, PFtoGoC, GoCtoGoC, GoCtoGoCgap, GlotoGrC, AAtoMLI, PFtoMLI

Usage:
  run_connector.py (-i PATH) (-o PATH) (-p PATH) [--parallel] (all | <jobs>...)
  run_connector.py (-h | --help)
  run_connector.py --version

Options:
  -h --help                            Show this screen
  --version                            Show version
  -i PATH, --input_path=<input_path>   Input path
  -o PATH, --output_path=<output_path> Output path
  -p PATH, --param_path=<param_dir>    Params path
  --parallel                           Parallel run (via ipyparallel)

Written by Ines Wichert and Sungho Hong
Supervised by Erik De Schutter
Computational Neuroscience Unit,
Okinawa Institute of Science and Technology

March, 2020
"""

from pathlib import Path
import pycabnn as cbn
import time
import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.neighbors import KDTree
from pycabnn.util import Query_point as qp
from docopt import docopt


def load_input_data(args):
    data = {}
    for arg in args:
        if "path" in arg:
            data[arg.replace("--", "")] = Path(args[arg]).resolve()

    data["config_hoc"] = data["param_path"] / "Parameters.hoc"
    data["parallel"] = args["--parallel"]

    t1 = time.time()
    print("Starting parallel process...")

    # Read the config file
    from pycabnn.util import HocParameterParser

    h = HocParameterParser()
    config_hoc = str(data["config_hoc"])
    h.load_file(config_hoc)

    t2 = time.time()
    print("Import finished:", t2 - t1)
    data["t"] = t2
    data["h"] = h
    return data


def load_and_make_population(data, pops):

    output_path = data["output_path"]
    h = data["h"]

    def make_glo(data):
        """sets up the Glomerulus population"""
        from pycabnn.cell_population import Glo_pop

        glo_data = np.loadtxt(data["param_path"] / "GLpoints.dat")

        glo = Glo_pop(h)
        if glo_data.shape[1] == 4:
            glo.load_somata(glo_data[:, 1:])
            glo.mf = glo_data[:, 0]
        else:
            glo.load_somata(glo_data)
        glo.save_somata(output_path, "GLcoordinates.sorted.dat")

        t0 = data["t"]
        t1 = time.time()
        print("GL processing:", t1 - t0)
        return (glo, t1)

    def make_goc(data):
        """sets up the Golgi population, render dendrites."""
        gol_in = data["input_path"] / "GoCcoordinates.dat"
        gg = cbn.create_population("Golgi", h)
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
        gp = cbn.create_population("Granule", h)
        gp.load_somata(gran_in)
        gp.add_aa_endpoints_fixed()
        gp.add_pf_endpoints()
        gp.save_gct_points(output_path)
        gp.save_somata(output_path, "GCcoordinates.sorted.dat")

        t3 = data["t"]
        t4 = time.time()
        print("Granule cell processing:", t4 - t3)
        return (gp, t4)

    def make_MLI(data):
        """sets up the Golgi population, render dendrites."""
        mli_in = data["input_path"] / "MLIcoordinates.dat"
        mli = cbn.create_population("MLI", h)
        mli.load_somata(mli_in)

        mli.load_data(data["input_path"] / "MLIDendData.npz")
        # mli.add_axon()
        mli.save_somata(output_path, "MLIcoordinates.sorted.dat")

        t2 = data["t"]
        t3 = time.time()
        print("MLI processing:", t3 - t2)
        return (mli, t3)

    print(" ")

    data["pops"] = {}
    for c in pops:
        data["pops"][c], t = eval("make_" + c)(data)
        data["t"] = t

    return data


def run_AAtoGoC(data):
    h = data["h"]

    # you might want to change the radii
    c_rad_aa = h.AAtoGoCzone / 1.73
    print("R for AA: {}".format(c_rad_aa))

    gp, gg = data["pops"]["grc"], data["pops"]["goc"]
    cc = cbn.Connect_2D(gp.qpts_aa, gg.qpts, c_rad_aa)
    # TODO: adapt the following from command line options
    cc.connections_parallel(
        parallel=data["parallel"], nblocks=120, debug=False
    )
    cc.save_result(data["output_path"] / "AAtoGoC")

    t5 = time.time()
    t4 = data["t"]
    print("AA: Found and saved after", t5 - t4)
    print(" ")
    data["t"] = t5
    return data


def run_PFtoGoC(data):
    h = data["h"]

    c_rad_pf = h.PFtoGoCzone / 1.113
    print("R for PF: {}".format(c_rad_pf))

    gp, gg = data["pops"]["grc"], data["pops"]["goc"]
    cc = cbn.Connect_2D(gp.qpts_pf, gg.qpts, c_rad_pf)
    cc.connections_parallel(
        parallel=data["parallel"], nblocks=120, debug=False
    )
    cc.save_result(data["output_path"] / "PFtoGoC")

    t6 = time.time()
    t5 = data["t"]
    print("PF: Found and saved after", t6 - t5)
    print(" ")
    data["t"] = t6
    return data


def run_AAtoMLI(data):
    h = data["h"]

    # you might want to change the radii
    c_rad_aa = h.AAtoMLI_cdist # critical distance for AA and MLI
    print("R for AA: {}".format(c_rad_aa))

    gp, mli = data["pops"]["grc"], data["pops"]["MLI"]
    cc = cbn.Connect_2D(gp.qpts_aa, mli.dends, c_rad_aa) #mli.qpts
    # TODO: adapt the following from command line options
    cc.connections_parallel(
        parallel=data["parallel"], nblocks=120, debug=False
    )
    cc.save_result(data["output_path"] / "AAtoMLI")

    t5 = time.time()
    t4 = data["t"]
    print("AA: Found and saved after", t5 - t4)
    print(" ")
    data["t"] = t5


def run_PFtoMLI(data):
    h = data["h"]

    c_rad_pf = h.PFtoMLI_cdist
    print("R for PF: {}".format(c_rad_pf))

    gp, mli = data["pops"]["grc"], data["pops"]["MLI"]
    cc = cbn.Connect_2D(gp.qpts_pf, mli.dends, c_rad_pf) #mli.qpts
    cc.connections_parallel(
        parallel=data["parallel"], nblocks=120, debug=False
    )
    cc.save_result(data["output_path"] / "PFtoMLI")

    t6 = time.time()
    t5 = data["t"]
    print("PF: Found and saved after", t6 - t5)
    print(" ")
    data["t"] = t6
    return data


def run_GoCtoGoC(data):
    h = data["h"]
    output_path = data["output_path"]

    gg = data["pops"]["goc"]

    # Golgi cell axon points
    gax = qp(gg.axon)

    ii, di = KDTree(gax.coo).query_radius(
        gg.som, r=h.GoCtoGoCzone, return_distance=True
    )

    # Find all sources and exclude self-connections
    srcs = [np.unique(gax.idx[ix].ravel()) for ix in ii]
    srcs = [s[s != n] for n, s in enumerate(srcs)]

    # Compute the soma-axon distance
    axdst = gax.coo - gg.som[gax.idx.ravel(), :]
    axdst = np.sqrt(np.sum(axdst ** 2, axis=1))

    src = []
    tgt = []
    dist = []

    # Compute soma-axon-soma distance and collect data
    for n, _ in enumerate(srcs):
        for s in srcs[n]:
            idx_axon = gax.idx[ii[n]].ravel() == s
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
    h = data["h"]
    gg = data["pops"]["goc"]
    dist = []
    src = []
    tgt = []

    # Find all pairs within GoCtoGoCgapzone
    ii, di = KDTree(gg.som).query_radius(
        gg.som, r=h.GoCtoGoCgapzone, return_distance=True
    )

    srcs = [s[s != n] for n, s in enumerate(ii)]
    dists = [di[n][s != n] for n, s in enumerate(ii)]

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

    r_search = 7.85  # search for glos within this dist from each grc
    scale_factor = 1 / 4  # scale the sagittal coords by this factor

    grc = data["pops"]["grc"].som.copy()
    glo = data["pops"]["glo"].som.copy()

    grc[:, 1] *= scale_factor
    glo[:, 1] *= scale_factor

    ii, di = KDTree(grc).query_radius(glo, radius=r_search, return_distance=True)

    dist = []
    src = []
    tgt = []

    for i in range(grc.som.shape[0]):
        for n, j in enumerate(ii[i]):
            src.append(i)
            tgt.append(j)
            dist.append(di[n])

    output_path = data["output_path"]

    np.savetxt(output_path / "GLtoGCdistances.dat", dist)
    np.savetxt(output_path / "GLtoGCsources.dat", src, fmt="%d")
    np.savetxt(output_path / "GLtoGCtargets.dat", tgt, fmt="%d")


def main(args):
    data = load_input_data(args)
    data = load_and_make_population(data, ["grc", "MLI"])
    print(data)

    valid_job_list = [
        "AAtoGoC", 
        "PFtoGoC", 
        "GoCtoGoC", 
        "GoCtoGoCgap",
        "AAtoMLI", 
        "PFtoMLI"
    ]

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
    args = docopt(__doc__, version=cbn.__version__)
    main(args)
