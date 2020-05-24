"""Utility functions.

Written by Ines Wichert and Sungho Hong
Supervised by Erik De Schutter
Computational Neuroscience Unit,
Okinawa Institute of Science and Technology

March, 2020
"""

import numpy as np
from numpy import sqrt
from numpy import pi as PI


def str_l(ar):
    """make a space-seperated string from all elements in a 1D-array"""
    return " ".join(str(ar[i]) for i in range(len(ar)))


def print_time_and_reset(t, comment="Finished block after "):
    import time

    t_new = time.time()
    print(comment, t_new - t)
    return t_new


class Pseudo_hoc(object):
    """Up to now, the program depends on a hoc object that contains the parameters for the simulation.
    However, as in the cluster, neuron is not installed for python 3, this class is a workaround:
    First, a dict has to be generated(and probably pickled) from the hoc file in a python distribution that has neuron installed.
    This dict(or the file containing it) is then read in in this class, and an empty object with no other functionalities gets
    assigned all the parameters from the dict as attributes. The resulting object can then be used as a parameter carrier just as the hoc file."""

    def __init__(self, ad_or_fn=None):
        """Add parameters from dict or filename to pseudo_hoc object"""
        # Make a pseudo-hoc object
        if ad_or_fn is None:
            return
        elif type(ad_or_fn) != dict:
            try:
                ad_or_fn = Path(ad_or_fn)
                # print (ad_or_fn.exists())
                # print (ad_or_fn.is_file())
                # print (ad_or_fn.absolute())

                import pickle

                with ad_or_fn.open("rb") as f_in:
                    ad_or_fn = pickle.load(f_in)
            except:
                print("Tried to read in ", ad_or_fn, " as a file, failed")
                return
        else:
            assert type(ad_or_fn) == dict, "Could not read in {}".format(ad_or_fn)
        # Add all elements from the read in file as arguments to the pseudo-hoc object
        for k, v in ad_or_fn.items():
            # As for the pickling process, all values had to be declared as strings, try to convert them back to a number
            try:
                v = float(v)
            except:
                pass
            try:
                setattr(self, k, v)
            except:
                pass

    def convert_hoc_to_pickle(self, config_fn, output_fn="pseudo_hoc.pkl"):
        """Take a .hoc config file and pickle it as a neuron-independent python dict."""
        try:
            import neuron
        except ModuleNotFoundError:
            print(
                "Could not import neuron, go to a python environment with an installed neuron version and try again."
            )
            return
        neuron.h.xopen(config_fn)
        d = dir(h)
        h_dict = dict()
        # Transfer parameters from the hoc object to a python dictionary
        for n, el in enumerate(d):
            if el[0].isupper() or el[0].islower():
                try:
                    # The value has to be converted to its string representation to get rid of the hoc properties.
                    # Must be kept in mind when reading in though.
                    h_dict[el] = repr(getattr(h, el))
                except:
                    pass
        # Dump the dictionary
        import pickle

        with output_fn.open("wb") as f:
            pickle.dump(h_dict, f)


class Query_point(object):
    def __init__(
        self,
        coord,
        IDs=None,
        segs=None,
        lin_offset=0,
        set_0=0,
        prevent_lin=False,
        dists=None,
    ):
        """Make a Query_point object from a point array and any meta data:
        The coord array should have either the shape (#points, point dimension) or
        (#cells, #points per cell, point dimenstion). In the second case the array will be reshaped to be like the first case,
        with the additional attributes IDs (cell, first dimenion of coord), and segs (second dimension of coord).
        It will be automatically checked whether the points can be linearized/projected, i.e. represented by a start, end, and 2-D projection"""

        self.npts = len(coord)
        # check if lin -> then it can be used for the Connect_2D method.
        # In that case it will not be
        if not prevent_lin:
            self.lin = self.lin_check(coord)
            if self.lin:
                # lin_offset will be added to the distance for each connection (e.g. aa length for pf)
                try:
                    lin_offset = float(np.array(lin_offset)) * np.ones(self.npts)
                except:
                    assert (
                        len(lin_offset) == self.npts
                    ), "lin_offset should be a scalar or an array with length npts!"
                finally:
                    self.lin_offset = lin_offset
                # set0 sets where 0 is defined along the elongated structure (e.g. branching point for PF)
                try:
                    set_0 = float(np.array(set_0)) * np.ones(self.npts)
                except:
                    assert (
                        len(set_0) == self.npts
                    ), "lin_offset should be a scalar or an array with length npts!"
                finally:
                    self.set_0 = set_0

                self.coo = coord
                self.seg = np.ones(len(coord))
                if IDs is None:
                    self.idx = np.arange(len(coord))
                else:
                    assert len(IDs) == len(coord), "ID length does not match "
                    self.idx = IDs
                return
        self.lin = False

        # If the structure is already flattened or there is only one point per ID
        if len(coord.shape) == 2:
            self.coo = coord
            if IDs is not None:
                assert len(coord) == len(
                    IDs
                ), "Length of ID list and length of coordinate file must be equal"
                self.idx = IDs
            if segs is None:
                self.seg = np.ones(len(coord))
                if IDs is None:
                    self.idx = np.arange(len(coord))
            else:
                assert not np.all(
                    IDs == None
                ), "Cell IDs must be sepcified before segment number"
                self.seg = segs

        # If the input array still represents the structure
        if len(coord.shape) == 3:
            assert (IDs is None) == (
                segs is None
            ), "To avoid confusion, cell IDs and segment numbers must either be specified both, or neither"
            if (not (IDs is None)) and (not (segs is None)):
                assert np.all(IDs.shape == coord.shape[:-1]) and np.all(
                    segs.shape == coord.shape[:-1]
                ), "Dimensions of ID and segment file should be " + str(
                    coord.shape[:-1]
                )
            else:
                IDs = np.array(
                    [
                        [[i] for j in range(coord.shape[1])]
                        for i in range(coord.shape[0])
                    ]
                )
                segs = np.array(
                    [
                        [[j] for j in range(coord.shape[1])]
                        for i in range(coord.shape[0])
                    ]
                )
            lam_res = lambda d: d.reshape(d.shape[0] * d.shape[1], d.shape[2])
            self.coo = lam_res(coord)
            self.seg = lam_res(np.expand_dims(segs, axis=2))
            self.idx = lam_res(np.expand_dims(IDs, axis=2))

    # check if input array can be used for the projection method (Connect_2D) (-> the points have a start and an end point)
    # Note that a lot of points describing a line will not be projected automatically
    def lin_check(self, coord):
        if len(coord.shape) == 3:
            if coord.shape[1] == 2 and coord.shape[2] == 3:
                sm = sum(abs(coord[:, 0, :] - coord[:, 1, :]))
                no_dif = [np.isclose(sm[i], 0) for i in range(len(sm))]
                if sum(no_dif) == 2:  # 2 coordinates are the same, one is not
                    self.lin_axis = np.invert(
                        no_dif
                    )  # this one is the axis that cn be linearized
                    return True
        return False

    def linearize(self):
        pass
        # this function should linearize points when they are in a higher structure than nx3, and the IDs and


class HocParameterParser(object):
    def load_file(self, hoc_file_path):
        with open(hoc_file_path) as f:
            for line in f.readlines():
                non_comment = line.split("//")[0]
                if "=" in non_comment:
                    # print(f"Found non-comment: {non_comment}")
                    exec(non_comment)
                    key = non_comment.split("=")[0].strip()
                    exec(f"self.{key}={key}")
                # if non_comment == "":
                #     print(f"Whole comment: {line.strip()}")


if __name__ == "__main__":
    h = HocParameterParser()
    h.load_file("../test_data/params/Parameters.hoc")
    for k in h.__dict__:
        print(f"{k}: {h.__dict__[k]}")
