#!/usr/bin/env python
"""generate_cell_position.py

Jobs = mf, goc, glo, grc

Usage:
  generate_cell_position.py (-o PATH) (-p PATH) [-i PATH] (all | <jobs>...)
  generate_cell_position.py (-h | --help)
  generate_cell_position.py --version

Options:
  -h --help                            Show this screen.
  --version                            Show version.
  -o PATH, --output_path=<output_path> Output path.
  -p PATH, --param_path=<param_dir>    Params path.
  -i PATH, --input_path=<input_path>   Input path.

"""

import numpy as np
from pycabnn.pop_generation.ebeida import ebeida_sampling
from pycabnn.pop_generation.utils import PointCloud

valid_job_list = ["mf", "goc", "glo", "grc"]


def load_input_data(args):
    from neuron import h
    from pathlib import Path

    param_file = Path(args["--param_path"]) / "Parameters.hoc"
    h.load_file(str(param_file))

    # Limit the x-range to 700 um and add 50 um in all directions
    h.MFxrange = 700
    h.MFxrange += 50
    h.MFyrange += 50
    h.GLdepth += 50

    foutname = Path(args["--output_path"])
    if foutname.suffix != ".npz":
        foutname = foutname.with_suffix(".npz")

    data = {"h": h, "foutname": foutname}

    if args["--input_path"]:
        existing_data = np.load(args["--input_path"])
        for c in existing_data:
            data[c + "_points"] = existing_data[c]

    return data


def make_mf(data):
    h = data["h"]
    foutname = data["foutname"]

    def compute_mf_params():
        Transverse_range = h.MFyrange
        Horizontal_range = h.MFxrange
        Vertical_range = h.GLdepth
        Volume = Transverse_range * Horizontal_range * Vertical_range

        MFdensity = h.MFdensity

        box_fac = 2.5
        Xinstantiate = 64 + 40  # 297+40
        Yinstantiate = 84 + 40 * box_fac  # 474+40x*box_fac

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

    spacing_mf = 20.9

    mf_points = ebeida_sampling(mf_box, spacing_mf, n_mf, True)

    data["mf_points"] = mf_points
    np.savez(foutname, mf=mf_points)

    return data


def make_goc(data):
    h = data["h"]
    foutname = data["foutname"]

    def compute_goc_params():
        Transverse_range = h.MFyrange
        Horizontal_range = h.MFxrange
        Vertical_range = h.GLdepth + 50

        Volume = Transverse_range * Horizontal_range * Vertical_range

        d_goc = h.GoCdensity
        n_goc = int(d_goc * Volume * 1e-9)
        print("Target # GoCs = {}".format(n_goc))
        return ((Horizontal_range, Transverse_range, Vertical_range), n_goc)

    goc_box, n_goc = compute_goc_params()

    spacing_goc = 45 - 1  # 40 (NH Barmack, V Yakhnitsa, 2008)

    goc_points = ebeida_sampling(goc_box, spacing_goc, n_goc, True)

    goc_points = goc_points + np.random.normal(
        0, 1, size=(len(goc_points), 3)
    )  # Gaussian noise

    data["goc_points"] = goc_points
    np.savez(foutname, mf=data["mf_points"], goc=goc_points)

    return data


def make_glo(data):
    h = data["h"]
    foutname = data["foutname"]
    goc_points = data["goc_points"]

    # (Billings et al., 2014) Since glomeruli is elipsoid shape, I recalculated based on the spatial occupancy of glomeruli and its density. Also, I subtract 1 cuz I will give Gaussian noise
    scale_factor = 1 / 3
    spacing_glo = 8.39 - 1
    d_goc_glo = 27 / 2 + spacing_glo / 2 - 1 + 1 / scale_factor

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

        d_grc = 1.9 * 1e6  # (Billings et al., 2014)
        d_glo = d_grc * 0.3
        #     d_glo = 6.6 * 1e5  # (Billings et al., 2014)
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

    # Glomerulus (Rosettes)
    globox, n_glo = compute_glo_params()

    glo_points = ebeida_sampling(globox, spacing_glo, n_glo, True, ftests=[goc])

    # Since glomerulus is stretched in a sagittal direction, we will generate coordinates in small area at first, and then multiply it with 3. (Billings et al., 2014)
    # glo_points[:, 1] = glo_points[:, 1] / scale_factor
    # glo_points1 = glo_points.copy()
    # glo_points1[:, 1] = glo_points1[:, 1] * scale_factor
    glo_points1 = glo_points + np.random.normal(0, 1, size=glo_points.shape)
    glo_points1[:, 1] = glo_points1[:, 1] / scale_factor

    data["glo_points"] = glo_points1

    np.savez(
        foutname, mf=data["mf_points"], goc=data["goc_points"], glo=data["glo_points"]
    )

    return data


def make_grc(data):
    h = data["h"]
    foutname = data["foutname"]

    goc_points = data["goc_points"]
    glo_points = data["glo_points"]

    d_goc_grc = 27 / 2 + 6.15 / 2
    goc = PointCloud(goc_points, d_goc_grc)

    d_glo_grc = 8.39 / 2 + 6.15 / 2
    glo = PointCloud(glo_points, d_glo_grc)

    def compute_grc_params():
        Transverse_range = h.MFyrange
        Horizontal_range = h.MFxrange
        Vertical_range = h.GLdepth
        Volume = Transverse_range * Horizontal_range * Vertical_range

        d_grc = 1.9 * 1e6  # (Billings et al., 2014)
        n_grc = int(d_grc * Volume * 1e-9)

        print("Target # GrCs = {}".format(n_grc))

        return ((Horizontal_range, Transverse_range, Vertical_range), n_grc)

    spacing_grc = 6.15 - 0.2

    grcbox, n_grc = compute_grc_params()
    grc_points = ebeida_sampling(grcbox, spacing_grc, n_grc, True, ftests=[glo, goc])

    data["grc_points"] = grc_points

    np.savez(
        foutname,
        mf=data["mf_points"],
        goc=data["goc_points"],
        glo=data["glo_points"],
        grc_nop=grc_points,
    )

    return data


def main(args):
    data = load_input_data(args)

    if args["all"]:
        args["<jobs>"] = valid_job_list

    for j in args["<jobs>"]:
        if j not in valid_job_list:
            raise RuntimeError(
                "Job {} is not valid, not in {}".format(j, valid_job_list)
            )
        else:
            data = eval("make_" + j)(data)


if __name__ == "__main__":
    from docopt import docopt

    args = docopt(__doc__, version="0.7dev")
    main(args)
