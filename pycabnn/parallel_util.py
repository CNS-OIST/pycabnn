"""parallel_util.py

contains functions that will be imported to each worker in the parallel version of pycabnn. Queries given points in a given tree, saves results.

Written by Ines Wichert and Sungho Hong
Supervised by Erik De Schutter
Computational Neuroscience Unit,
Okinawa Institute of Science and Technology

March, 2020
"""

import pandas as pd
from .util import str_l


def find_connections_2dpar(
    kdt, pts, lpts, c_rad, lin_axis, lin_in_tree, lin_is_src, ids, debug=False
):
    """
    Performs distance-based searches for the linearized version of pycabnn (currently Connect_2D)
    kdt: 2D Tree with points
    pts: Query_point object for the 3D data points
    lpts: Query_point object for the 2D projected points
    c_rad: critical radius, i.e. maximum distance between points to consider them connected
    lin_axis: axis along which the data is projected
    lin_in_tree: whether the linear structure is in the tree or not
    lin_is_src: whether the linear structure should be saved in the source file
    ids: IDs of the query points that this particular worker should deal with
    prefix: Path and prefix of the files under which the worker should save the data
    """

    import numpy

    if lin_in_tree:
        q_pts = pts.coo[:, numpy.invert(lin_axis)]
    else:
        q_pts = lpts.coo[:, 0, numpy.invert(lin_axis)]

    lax_c = pts.coo[:, lin_axis]
    if lin_in_tree:
        lax_c = lax_c[ids[1]]  # gives correct coordinates via correct indexing
    lax_range = lpts.coo[:, :, lin_axis]
    lax_range = lax_range.reshape((lax_range.shape[0], lax_range.shape[1]))

    res = []
    res_l = []

    # iterate through the query points
    for i, pt in enumerate(q_pts[ids[1]]):
        # find the points within the critical radius
        (ind,) = kdt.query_radius(numpy.expand_dims(pt, axis=0), r=c_rad)

        # check if the found points match along the linearized axis and if so, add distance from the beginning of the linearized axis
        if lin_in_tree:
            ind = ind[
                numpy.logical_and(
                    lax_range[ind, 0] <= lax_c[i], lax_range[ind, 1] >= lax_c[i]
                )
            ]
            res_l.append(
                abs(lax_c[i] - lax_range[ind, 0] - lpts.set_0[ind])
                + lpts.lin_offset[ind]
            )
        else:
            ind = ind[
                numpy.logical_and(
                    lax_range[i, 0] <= lax_c[ind], lax_range[i, 1] >= lax_c[ind]
                ).ravel()
            ]
            res_l.append(
                abs(lax_c[ind].ravel() - lax_range[i, 0] - lpts.set_0[i])
                + lpts.lin_offset[i]
            )

        res.append(ind.astype("int"))

    dfs = []
    for (l, cl, cl_l) in zip(list(ids[1]), res, res_l):

        assert len(cl) == len(
            cl_l
        ), "Something went wrong, all corresponding lists in your input arguments should have the same length"

        if len(cl_l) > 0:
            dist_data = cl_l
            # first, get the cell IDS of the query and tree points (for the linear points, that is just the point ID,
            # for the other points this information has to be extracted from the corresponding Query_points object.
            # Segments also corresponds to the 3D point population, right value is acquired from Query-points object.
            if lin_in_tree:
                s_ar = pts.seg[l, :].astype("int")
                seg_data = [s_ar for i in range(len(cl))]

                q_id = (numpy.ones(len(cl)) * pts.idx[l]).astype("int")
                tr_id = cl
            else:
                seg_data = pts.seg[cl].astype("int")
                # f_segs.write("\n".join(map(str_l, seg_data)))
                q_id = pts.idx[cl].ravel()
                tr_id = (numpy.ones(len(cl)) * l).astype("int")
            seg_data = numpy.array(seg_data)

            # depending on which population should be source and which should be target, save cell IDs accordingly.
            # if l == 1: print (lin_in_tree, lin_is_src)
            if lin_is_src:
                src_data = tr_id
                tgt_data = q_id
            else:
                src_data = q_id
                tgt_data = tr_id

            df = pd.DataFrame()
            df["source"] = src_data
            df["target"] = tgt_data
            df["segment"] = seg_data[:, 0]
            df["branch"] = seg_data[:, 1]
            df["distance"] = dist_data

            if debug:
                df["x"] = pts.coo[l][0]
                df["y"] = pts.coo[l][1]
                df["z"] = pts.coo[l][2]

            dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)

    return dfs


def find_connections_3dpar(
    kdt, spts, tpts, c_rad, src_in_tree, ids, use_distance, avoid_self
):
    """
    Performs distance-based searches between two populations of 3 dimensional point cloud structures
    spts: Query_point object for the source structure
    tpts: Query_point object for the target structure
    c_rad: critical radius, i.e. maximum distance between points to consider them connected
    lin_axis: axis along which the data is projected
    src_in_tree: whether the source population is in the tree or not
    ids: IDs of the query points that this particular worker should deal with
    prefix: Path and prefix of the files under which the worker should save the data
    """
    import numpy

    if src_in_tree:
        q_pts = tpts.coo
    else:
        q_pts = spts.coo

    res = []
    res_l = []
    # iterate through the query points
    for i, pt in enumerate(q_pts[ids[1]]):
        # find the points within the critical radius
        (ind,) = kdt.query_radius(numpy.expand_dims(pt, axis=0), r=c_rad)
        res.append(ind.astype("int"))
    dfs = []

    for (l, cl) in zip(list(ids[1].astype("int")), res):
        if len(cl) > 0:
            # depending on which population should be source and which should be target, save cell IDs accordingly.
            if src_in_tree:
                tgt_data = numpy.ones(len(cl)) * tpts.idx[l]
                src_data = spts.idx[cl.astype("int")]
                seg_data = [tpts.seg[l, :].astype("int") for i in range(len(cl))]
            else:
                tgt_data = tpts.idx[cl]
                src_data = numpy.ones(len(cl)) * spts.idx[l]
                seg_data = tpts.seg[cl.astype("int")]

    dfs = pd.concat(dfs, ignore_index=True)
    return dfs
