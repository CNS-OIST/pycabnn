#!/usr/bin/env python
"""generate_cell_position.py

Jobs = mf, goc, glo, grc

Usage:
  generate_cell_position.py (-o PATH) (-p PATH) [-i PATH] [--stop_method=<conditions>] (all | <jobs>...)
  generate_cell_position.py (-h | --help)
  generate_cell_position.py --version

Options:
  -h --help                               Show this screen
  --version                               Show version
  -o PATH, --output_path=<output_path>    Output path
  -p PATH, --param_path=<param_dir>       Params path
  -i PATH, --input_path=<input_path>      Input path
  --stop_method=<conditions>              Stop method. See below.

With <conditions>, you can specify how each job will stop. For example,

    `--stop_method=grc:maximal,goc:density`

will make the grc and goc generation use the maximal volume filling and density-based
stopping criterion, respectively. The default is the density-based, e.g., the job will
stop if the number of the cells reaches a target, computed from the given density.

Written by Sanghun Jee and Sungho Hong
Supervised by Erik De Schutter
Computational Neuroscience Unit,
Okinawa Institute of Science and Technology

March, 2020
"""

import numpy as np
import pycabnn
from pycabnn.util import HocParameterParser
from pycabnn.pop_generation.ebeida import ebeida_sampling
from pycabnn.pop_generation.utils import PointCloud

valid_job_list = ["mf", "goc", "glo", "grc", "mli"]


def load_input_data(args):
    from pathlib import Path

    param_file = Path(args["--param_path"]) / "Parameters.hoc"
    h = HocParameterParser()
    h.load_file(str(param_file))

    # Limit the x-range to 700 um and add 50 um in all directions
    h.MFxrange += h.range_margin
    h.MFyrange += h.range_margin
    h.GLdepth += h.range_margin

    output_path = Path(args["--output_path"])
    foutname = output_path / "cell_positions.npz"

    data = {"h": h, "foutname": foutname, "output_path": output_path}

    if args["--input_path"]:
        existing_data = np.load(args["--input_path"])
        for c in existing_data:
            data[c + "_points"] = existing_data[c]

    return data


def make_mf(data):
    h = data["h"]
    foutname = data["foutname"]
    stop_cond = data['stop_conds']['mf']

    def compute_mf_params():
        Transverse_range = h.MFyrange
        Horizontal_range = h.MFxrange
        Vertical_range = h.GLdepth
        Volume = Transverse_range * Horizontal_range * Vertical_range

        MFdensity = h.MFdensity

        Xinstantiate = h.MFxextent
        Yinstantiate = h.MFyextent

        n_mf = int(
            (Transverse_range + (2 * Xinstantiate))
            * (Horizontal_range + (2 * Yinstantiate))
            * MFdensity
            * 1e-6
        )

        print("Target # MFs = {}".format(n_mf))
        return (
            (
                Horizontal_range + (2 * Yinstantiate),
                Transverse_range + (2 * Xinstantiate),
            ),
            n_mf,
        )

    mf_box, n_mf = compute_mf_params()
    mf_points = ebeida_sampling(mf_box, h.spacing_mf, n_mf, True, stop_method=stop_cond)

    data["mf_points"] = mf_points

    print("Final # MFs = {}".format(mf_points.shape[0]))
    
    np.savez(foutname, mf=mf_points)
    np.savetxt(data["output_path"] / "MFcoordinates.dat", data["mf_points"])

    return data


def make_goc(data):
    h = data["h"]
    foutname = data["foutname"]
    stop_cond = data['stop_conds']['goc']

    def compute_goc_params():
        Transverse_range = h.MFyrange
        Horizontal_range = h.MFxrange
        Vertical_range = h.GLdepth

        Volume = Transverse_range * Horizontal_range * Vertical_range

        d_goc = h.GoCdensity
        n_goc = int(d_goc * Volume * 1e-9)
        print("Target # GoCs = {}".format(n_goc))
        return ((Horizontal_range, Transverse_range, Vertical_range), n_goc)

    goc_box, n_goc = compute_goc_params()

    spacing_goc = h.spacing_goc - h.softness_margin_goc

    goc_points = ebeida_sampling(goc_box, spacing_goc, n_goc, True, stop_method=stop_cond)

    goc_points = (
        goc_points
        + np.random.normal(0, 1, size=(len(goc_points), 3)) * h.softness_margin_goc
    )  # Gaussian noise

    data["goc_points"] = goc_points

    print("Final # GoCs = {}".format(goc_points.shape[0]))

    np.savez(foutname, mf=data["mf_points"], goc=goc_points)
    np.savetxt(data["output_path"] / "GoCcoordinates.dat", data["goc_points"])

    return data


def make_glo(data):
    h = data["h"]
    foutname = data["foutname"]
    stop_cond = data['stop_conds']['glo']
    goc_points = data["goc_points"]

    scale_factor = h.scale_factor_glo
    spacing_glo = h.diam_glo - h.softness_margin_glo

    # minimal distance between GoCs and glomeruli
    # softness margin is applied only to glo since only the glo coords will be
    # perturbed
    d_goc_glo = h.diam_goc / 2 + h.diam_glo / 2 - h.softness_margin_glo

    class GoC(PointCloud):
        def test_points(self, x):
            y = x.copy()
            y[:, 1] = y[:, 1] / scale_factor
            return super().test_points(y)

        def test_cells(self, cell_corners, dgrid, nn=None):
            y = cell_corners.copy()
            y[:, 1] = y[:, 1] / scale_factor
            return super().test_cells(y, dgrid, nn=nn)

    goc = GoC(goc_points, d_goc_glo)
    goc.dlat[:, 1] = goc.dlat[:, 1] / scale_factor

    def compute_glo_params():
        Transverse_range = h.MFyrange
        Horizontal_range = h.MFxrange
        Vertical_range = h.GLdepth
        Volume = Transverse_range * Horizontal_range * Vertical_range

        d_glo = h.density_glo
        n_glo = int(d_glo * Volume * 1e-9)
        print("Target # Glomeruli = {}".format(n_glo))

        return (
            (
                Horizontal_range,
                int(Transverse_range * scale_factor + 0.5),
                Vertical_range,
            ),
            n_glo,
        )

    globox, n_glo = compute_glo_params()

    glo_points = ebeida_sampling(globox, spacing_glo, n_glo, True, ftests=[goc],stop_method=stop_cond)

    # Since the glomerulus distribution is stretched in a sagittal direction,
    # we generate the coordinates in a scaled volume first, and then stretch
    # the sagittal coordinate by 3
    glo_points1 = (
        glo_points
        + np.random.normal(0, 1, size=glo_points.shape) * h.softness_margin_glo
    )
    glo_points1[:, 1] = glo_points1[:, 1] / scale_factor

    data["glo_points"] = glo_points1

    print("Final # Glos = {}".format(glo_points1.shape[0]))

    np.savez(
        foutname, mf=data["mf_points"], goc=data["goc_points"], glo=data["glo_points"]
    )
    np.savetxt(data["output_path"] / "GLpoints.dat", data["glo_points"])

    return data


def make_grc(data):
    h = data["h"]
    foutname = data["foutname"]
    stop_cond = data['stop_conds']['grc']

    goc_points = data["goc_points"]
    glo_points = data["glo_points"]

    # minimal distance between gocs and grcs
    d_goc_grc = h.diam_goc / 2 + h.diam_grc / 2 - h.softness_margin_grc
    goc = PointCloud(goc_points, d_goc_grc)

    # minimal distance between glomeruli and grcs
    d_glo_grc = h.diam_glo / 2 + h.diam_grc / 2 - h.softness_margin_grc
    glo = PointCloud(glo_points, d_glo_grc)

    def compute_grc_params():
        Transverse_range = h.MFyrange
        Horizontal_range = h.MFxrange
        Vertical_range = h.GLdepth
        Volume = Transverse_range * Horizontal_range * Vertical_range

        d_grc = h.density_grc
        n_grc = int(d_grc * Volume * 1e-9)

        print("Target # GrCs = {}".format(n_grc))

        return ((Horizontal_range, Transverse_range, Vertical_range), n_grc)

    spacing_grc = h.diam_grc - h.softness_margin_grc

    grcbox, n_grc = compute_grc_params()
    grc_points = ebeida_sampling(grcbox, spacing_grc, n_grc, True, ftests=[glo, goc],stop_method=stop_cond)

    grc_points = np.random.normal(0, 1, size=grc_points.shape) * h.softness_margin_grc

    data["grc_points"] = grc_points

    print("Final # GrCs = {}".format(grc_points.shape[0]))

    np.savez(
        foutname,
        mf=data["mf_points"],
        goc=data["goc_points"],
        glo=data["glo_points"],
        grc=data["grc_points"],
    )
    np.savetxt(data["output_path"] / "GCcoordinates.dat", data["grc_points"])

    return data

def make_mli(data):
    h = data["h"]
    foutname = data["foutname"]
    stop_cond = data['stop_conds']['goc']

    def compute_mli_params():
        Transverse_range = h.MFyrange
        Horizontal_range = h.MFxrange
        Vertical_range = h.MLdepth

        Volume = Transverse_range * Horizontal_range * Vertical_range

        d_mli = h.mli_density
        n_mli = int(d_mli * Volume * 1e-9)
        print("Target # MLIs = {}".format(n_mli))
        return ((Horizontal_range, Transverse_range, Vertical_range), n_mli)

    mli_box, n_mli = compute_mli_params()

    spacing_mli = h.spacing_mli - h.softness_margin_mli

    mli_points = ebeida_sampling(mli_box, spacing_mli, n_mli, True, stop_method=stop_cond)

    mli_points = (
        mli_points
        + np.random.normal(0, 1, size=(len(mli_points), 3)) * h.softness_margin_mli
    )  # Gaussian noise

    data["mli_points"] = mli_points

    print("Final # MLIs = {}".format(mli_points.shape[0]))

    np.savez(foutname, mli=mli_points)
    np.savetxt(data["output_path"] / "MLIcoordinates.dat", data["mli_points"])

    return data


def main(args):
    data = load_input_data(args)
    
    if args["all"]:
        args["<jobs>"] = valid_job_list

    if args['--stop_method']:
        conds = args['--stop_method'].split(',')
        conds = [c.split(':') for c in conds]
        conds = [(x.strip(), y.strip()) for x, y in conds]
        data['stop_conds'] = dict(conds)

    for j in args["<jobs>"]:

        if j not in data['stop_conds']:
            data['stop_conds'][j] = 'density'

        if j not in valid_job_list:
            raise RuntimeError(
                "Job {} is not valid, not in {}".format(j, valid_job_list)
            )
        else:
            data = eval("make_" + j)(data)

if __name__ == "__main__":
    from docopt import docopt

    args = docopt(__doc__, version=pycabnn.__version__)
    main(args)
