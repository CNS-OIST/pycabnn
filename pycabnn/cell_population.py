"""cell_population

Written by Ines Wichert, Sanghun Jee, and Sungho Hong
Supervised by Erik De Schutter
Computational Neuroscience Unit,
Okinawa Institute of Science and Technology

March, 2020
"""

import numpy as np
from pathlib import Path
import csv
from .util import str_l, Query_point
from tqdm import tqdm, trange


class Cell_pop(object):
    def __init__(self, my_args):
        self.args = my_args

    def load_somata(self, fn_or_ar):
        """ Loads somata and adds them to the population object:
        Either from a coordinate file(if fn_or_ar is a filename)
        Or diectly from a numpy array(if fn_or_ar is such)"""
        if hasattr(fn_or_ar, "shape"):
            try:
                if fn_or_ar.shape[len(fn_or_ar.shape) - 1] == 3:
                    self.som = fn_or_ar
                else:
                    print("Cannot recognize array as coordinates")
            except:
                print("Could not read in soma points")
        else:
            try:
                self.read_in_soma_file(fn_or_ar)
                print("Successfully read {}.".format(fn_or_ar))
            except:
                print("Tried to read in ", fn_or_ar, ", failed.")

    def save_somata(self, prefix="", fn=""):
        """Save the soma coordinates"""
        prefix = Path(prefix)
        if fn == "":
            fn = type(self).__name__ + "_coords.dat"

        assert hasattr(
            self, "som"
        ), "Cannot save soma coordinates, as apparently none have been added yet"

        with (prefix / fn).open("w") as f_out:
            f_out.write("\n".join(map(str_l, self.som)))
        print("Successfully wrote {}.".format(prefix / fn))

    def read_in_soma_file(self, fn, parse_ignore=True):
        """ Reads in files such as the ones that BREP returns.
        Represents lines as rows, nth element in each line as column.
        fn = Filename
        parse_ignore: If something cannot be parsed, it will be ignored. If this parameter is set false, it will complain
        returns: 2d-array of floats"""

        res = []
        with open(fn, "r", newline="") as f:
            rr = csv.reader(f, delimiter=" ")
            err = []  # list of elements that could not be read in
            for line in rr:  # lines -> rows
                ar = []
                for j in range(len(line)):
                    try:
                        ar.append(float(line[j]))
                    except:
                        err.append(line[j])
                res.append(np.asarray(ar))
        if len(err) > 0 and not parse_ignore:
            print("Could not parse on {} instances: {}".format(len(err), set(err)))
        self.som = np.asarray(res)
        self.n_cell = len(self.som)

    def coord_reshape(dat, n_dim=3):
        """ Reshapes coordinate files with several points in one line by adding an extra axis.
        Thus, converts from an array with shape(#cells x(#pts*ndim)) to one with shape(#cells x #pts x ndim)"""
        dat = dat.reshape([dat.shape[0], int(dat.shape[1] / n_dim), n_dim])
        return dat

    def gen_random_cell_loc(self, n_cell, Gc_x=-1, Gc_y=-1, Gc_z=-1, sp_std=2):
        """Random generation for cell somatas
        n_cell(int) = number of cells
        Gc_x, Gc_y, Gc_z(float) = dimensions of volume in which cells shall be distributed
        Algorithm will first make a grid that definitely has more elements than n_cell
        Each grid field is populated by a cell, then those cells are displaced randomly
        Last step is to prune the volume, i.e. remove the most outlying cells until the goal number of cells is left
        Returns: cell coordinates"""
        # get spacing for grid:
        if Gc_x < 0:
            Gc_x = self.args.GoCxrange
        if Gc_y < 0:
            Gc_y = self.args.GoCyrange
        if Gc_z < 0:
            Gc_z = self.args.GoCzrange

        vol_c = (
            Gc_x * Gc_y * Gc_z / n_cell
        )  # volume that on average contains exactly one cell
        sp_def = (
            vol_c ** (1 / 3) / 2
        )  # average spacing between cells(cube of above volume)

        # Get grid with a few too many elements
        gr = np.asarray(
            [
                [i, j, k]
                for i in np.arange(0, Gc_x, 2 * sp_def)
                for j in np.arange(0, Gc_y, 2 * sp_def)
                for k in np.arange(0, Gc_z, 2 * sp_def)
            ]
        )
        # random displacement
        grc = gr + np.random.randn(*gr.shape) * sp_def * sp_std

        # then remove the ones that lie most outside to get the correct number of cells:
        # First find the strongest outliers
        lower = grc.T.ravel()
        upper = -(grc - [Gc_x, Gc_y, Gc_z]).T.ravel()
        most_out_idx = np.mod(np.argsort(np.concatenate((lower, upper))), len(grc))
        # In order to find the right number, must iterate a bit as IDs may occur twice(edges)
        del_el = len(grc) - n_cell  # number of elements to be deleted
        n_del = del_el
        while len(np.unique(most_out_idx[:n_del])) < del_el:
            n_del = n_del + del_el - len(np.unique(most_out_idx[:n_del]))
        # Deletion step
        grc = grc[np.setdiff1d(np.arange(len(grc)), most_out_idx[:n_del]), :]

        # Now, this might still bee too far out, so we shrink or expand it.
        mi = np.min(grc, axis=0)
        ma = np.max(grc, axis=0)
        dim = [Gc_x, Gc_y, Gc_z]
        grc = (grc - mi) / (ma - mi) * dim

        self.som = grc
        return grc


class Golgi_pop(Cell_pop):
    """Golgi cell population. Generates point representations of axons and dendrites as well as Query_point objects from them"""

    def __init__(self, my_args):
        Cell_pop.__init__(self, my_args)

    def add_axon(self):
        """Adds axons as points with a uniform random distribution in a certain rectangle
        The seg array will contain the distance of the axon point from the soma"""
        x_r = [self.args.GoC_Axon_Xmin, self.args.GoC_Axon_Xmax]
        y_r = [self.args.GoC_Axon_Ymin, self.args.GoC_Axon_Ymax]
        z_r = [self.args.GoC_Axon_Zmin, self.args.GoC_Axon_Zmax]
        n_ax = int(self.args.numAxonGolgi)

        ar = np.random.uniform(size=[len(self.som), n_ax + 1, 3])
        for i, [low, high] in enumerate([x_r, y_r, z_r]):
            ar[:, :, i] = ar[:, :, i] * (high - low) + low
        ar[:, 0, :] = ar[:, 0, :] * 0
        for i in range(len(ar)):
            ar[i, :, :] = ar[i, :, :] + self.som[i, :]
        dists = np.linalg.norm(ar, axis=2)
        idx = np.array([[j for k in range(len(ar[j]))] for j in range(len(ar))])
        self.axon = ar
        self.axon_q = Query_point(ar, idx, segs=dists)  # TODO: this is weird

    def save_axon_coords(self, prefix=""):
        """ Save the coordinates of the dendrites, BREP style
        -> each line of the output file corresponds to one cell, and contains all its dendrites sequentially"""

        assert hasattr(self, "axon"), "Could not find axon, please generate first"

        prefix = Path(prefix)
        axon_file = prefix / "GoCaxoncoordinates_test.dat"
        with axon_file.open("w") as f_out:
            for ax in self.axon:
                flad = np.array([a for l in ax for a in l])
                f_out.write(str_l(flad) + "\n")
            print("Successfully wrote {}.".format(axon_file))

    def add_dendrites(self):
        """Add apical and basolateral dendrites using the parameters specified in the Parameters file.
        Will construct Query points by itself and add them to the object"""
        # apical
        a_rad = self.args.GoC_PhysApicalDendR  # radius of cylinder
        a_h = self.args.GoC_PhysApicalDendH  # height of cylinder
        a_ang = [
            self.args.GoC_Atheta_min,
            self.args.GoC_Atheta_max,
        ]  # mean angles for dendrite direction
        a_std = self.args.GoC_Atheta_stdev  # std for dendrite diredtion
        a_n = int(
            self.args.GoC_Ad_nseg * self.args.GoC_Ad_nsegpts
        )  # number of points per dendrite

        # basolateral
        b_rad = self.args.GoC_PhysBasolateralDendR
        b_h = self.args.GoC_PhysBasolateralDendH
        b_ang = [self.args.GoC_Btheta_min, self.args.GoC_Btheta_max]
        b_std = self.args.GoC_Btheta_stdev
        b_n = int(self.args.GoC_Bd_nseg * self.args.GoC_Bd_nsegpts)

        # generate the dendrite coordinates
        a_dend, a_idx, a_sgts = self.gen_dendrite(a_rad, a_h, a_ang, a_std, a_n)
        b_dend, b_idx, b_sgts = self.gen_dendrite(b_rad, b_h, b_ang, b_std, b_n)

        # The apical dendrites have higher numbers than the basal ones:
        a_sgts[:, :, 1] = a_sgts[:, :, 1] + len(
            b_ang
        )  # TODO: should add the number of basal dendrites instead
        # Taking into account that there are several points per segment
        # and the first segment has index 1
        a_sgts[:, :, 0] = np.floor(a_sgts[:, :, 0] / self.args.GoC_Ad_nsegpts) + 1
        b_sgts[:, :, 0] = np.floor(b_sgts[:, :, 0] / self.args.GoC_Bd_nsegpts) + 1

        # special concatenation function for apical and basal dendrite
        # stack coords/segments of a and b dends when they are from the same cell
        conc_ab_one = lambda i, a, b: np.vstack((a[a_idx == i], b[b_idx == i]))
        # loop around all the cells
        conc_ab = lambda a, b: np.vstack(
            conc_ab_one(i, a, b) for i in range(self.n_cell)
        )

        # concatenated dendrite information(coords, cell indices, segment information)
        all_dends = conc_ab(a_dend, b_dend)
        all_sgts = conc_ab(a_sgts, b_sgts)

        # put a and b indices together and rearrange them in a row
        all_idx = np.hstack((a_idx, b_idx))
        all_idx = all_idx.reshape((np.prod(all_idx.shape), 1))

        # test code for the part above
        # zz = all_dends[all_idx.flatten()==2,:]
        # plt.plot(zz[:,1], zz[:,2],'.')

        self.a_dend = a_dend
        self.b_dend = b_dend
        self.qpts = Query_point(all_dends, all_idx, all_sgts)

    def save_dend_coords(self, prefix=""):
        """ Save the coordinates of the dendrites, BREP style
        -> each line of the output file corresponds to one cell, and contains all its dendrites sequentially"""

        assert hasattr(self, "a_dend") or hasattr(
            self, "b_dend"
        ), "Could not find any added dendrites"

        prefix = Path(prefix)

        if hasattr(self, "a_dend"):
            dend_file = prefix / "GoCadendcoordinates.sorted.dat"
            with dend_file.open("w") as f_out:
                for ad in self.a_dend:
                    flad = np.array([a for l in ad for a in l])
                    f_out.write(str_l(flad) + "\n")
            print("Successfully wrote {}.".format(dend_file))
        else:
            warnings.warn("Could not find apical dendrite")

        if hasattr(self, "b_dend"):
            dend_file = prefix / "GoCbdendcoordinates.sorted.dat"
            with dend_file.open("w") as f_out:
                for bd in self.b_dend:
                    flbd = [b for l in bd for b in l]
                    f_out.write(str_l(flbd) + "\n")
            print("Successfully wrote {}.".format(dend_file))
        else:
            warnings.warn("Could not find basal dendrite")

    def gen_dendrite(self, c_r, c_h, c_m, c_std, c_n):
        """Generates dendrites as described in the paper:
        c_r = maximal radius of cone
        c_h = height of cone
        c_m = mean angle for each dendrite(number of elements = number of dendrites per cell)
        c_std = standard deviation(degree) for the angle of the dendrite
        c_n = number of points
        Returns three arrays:
        res: shape is #cells x #pts x 3(coords) -> coordinates of the points
        idx: shape is #cells x #pts -> cell ids of the points(starting at 0)
        sgts: shape is #cells x #pts x 2 -> each point consists of [# segment, # dendrite], both starting from 1
        -> to be conform with BREP, this has to be slightly modified, see add_dendrites function.
        where #pts = #segment per dendrite x# dendrites generated with this function
        """
        c_gr = np.linspace(0, 1, c_n) * np.ones(
            (3, c_n)
        )  # linspace grid between 0 and 1 with c_n elements
        b_res = []
        idx = []  # cell indices

        c_m = (
            -np.array(c_m) + 90
        )  # This is the angle conversion that is necessary to be compatible with the scheme BREP

        for i in range(len(self.som)):  # each cell
            som_c = self.som[i, :]
            d_res = []
            if i == 0:
                d_sg = (
                    []
                )  # segments, only have to be calculated once as they are the same for every cell
            for n, cc_m in enumerate(c_m):  # each dendrite
                ep_ang = (
                    (np.random.randn() * c_std + cc_m) * np.pi / 180
                )  # individual angle
                pt = (
                    [np.sin(ep_ang) * c_r, np.cos(ep_ang) * c_r, c_h]
                ) * c_gr.T  # coordinates of the dendrite = endpoint*grid
                d_res = d_res + list(pt + som_c)
                if i == 0:
                    d_sg = d_sg + list(
                        np.array([np.arange(c_n), np.ones(c_n) * (n + 1)]).T
                    )
            b_res.append(np.array(d_res))
            idx.append((np.ones(len(d_res)) * i).astype("int"))
        segs = np.array(
            [d_sg for k in range(i + 1)]
        )  # replicate segment information for each cell

        return np.array(b_res), np.array(idx), segs


class Granule_pop(Cell_pop):
    """Granule cell population. Generates point representations of ascending axon (aa) and parallel fiber(pf)
    Can either do so with 3D points or with 2D projections. AA length can be fixed or random"""

    def __init__(self, my_args):
        Cell_pop.__init__(self, my_args)
        self.aa_length = self.args.PCLdepth + self.args.GLdepth

    def add_aa_endpoints_random(self):
        """Generate aa endpoints with a random aa length(end point will be somewhere in mol_range)"""
        mol_range = [self.aa_length, self.aa_length + self.args.MLdepth]
        self.aa_dots = np.array(
            [np.array([self.som[i], self.som[i]]) for i in range(len(self.som))]
        )
        self.aa_dots[:, 1, 2] = np.random.uniform(
            mol_range[0], mol_range[1], len(self.aa_dots[:, 1, 2])
        )
        self.qpts_aa = Query_point(self.aa_dots)

    def add_aa_endpoints_fixed(self):
        """Generate aa endpoints with a fixed aa length"""
        # aa_length = self.args.PFzoffset   #NOTE: This value exists, but in the BREP original file it is replaced by the other definition
        self.aa_dots = np.array(
            [np.array([self.som[i], self.som[i]]) for i in range(len(self.som))]
        )
        self.aa_dots[:, 1, 2] = self.aa_dots[:, 1, 2] + self.aa_length
        self.qpts_aa = Query_point(self.aa_dots)

    def add_pf_endpoints(self):
        """ Add the endpoints of the parallel fibers [begin, end] for each cell"""
        pf_length = self.args.PFlength
        assert hasattr(
            self, "aa_dots"
        ), "Cannot add Parallel Fiber, add ascending axon first!"
        self.pf_dots = self.aa_dots.copy()
        self.pf_dots[:, 0, 2] = self.pf_dots[:, 1, 2]  # z axis shall be the same
        self.pf_dots[:, 0, 0] = self.pf_dots[:, 0, 0] - pf_length / 2
        self.pf_dots[:, 1, 0] = self.pf_dots[:, 1, 0] + pf_length / 2
        self.qpts_pf = Query_point(
            self.pf_dots,
            lin_offset=self.aa_dots[:, 1, 2] - self.aa_dots[:, 0, 2],
            set_0=pf_length / 2,
        )

    def add_3D_aa_and_pf(self):
        """adds 3-dimensional coordinates for ascending axons and parallel fiber to the granule cell objects.
        Both AA and PF are represented by regularly spaced dots"""

        aa_length = self.args.PCLdepth + self.args.GLdepth
        aa_nd = int(self.aa_length / self.args.AAstep)  # number of dots for the aa
        aa_sp = np.linspace(
            0, self.aa_length, aa_nd
        )  # grid that contains the spacing for the aa points

        pf_nd = int(self.args.PFlength / self.args.PFstep)  # number of dots for the pf
        pf_sp = np.linspace(
            -self.args.PFlength / 2, self.args.PFlength / 2, pf_nd
        )  # grid that contains spacing of po points

        self.aa_dots = np.zeros((len(self.som), aa_nd, 3))
        self.pf_dots = np.zeros((len(self.som), pf_nd, 3))
        aa_idx = np.zeros((len(self.som), aa_nd))
        aa_sgts = np.zeros((len(self.som), aa_nd))
        pf_idx = np.zeros((len(self.som), pf_nd))
        pf_sgts = np.zeros((len(self.som), pf_nd))

        for i, som in enumerate(self.som):

            self.aa_dots[i] = (
                np.ones((aa_nd, 3)) * som
            )  # copy soma location for each point of the aa
            self.aa_dots[i, :, 2] = self.aa_dots[i, :, 2] + aa_sp  # add the z offsets
            aa_idx[i, :] = i  # cell indices, for the query object
            aa_sgts[i, :] = np.arange(aa_nd)  # segment points, for the query object

            self.pf_dots[i] = (
                np.ones((pf_nd, 3)) * self.aa_dots[i, -1, :]
            )  # uppermost aa point is the origin of the pf points
            self.pf_dots[i, :, 0] = (
                self.pf_dots[i, :, 0] + pf_sp
            )  # this time, points differ only by their offset along the x direction
            pf_idx[i, :] = i
            pf_sgts[i, :] = np.arange(pf_nd)  #! Not necessarily nice

        self.qpts_aa = Query_point(self.aa_dots, aa_idx, aa_sgts)
        self.qpts_pf = Query_point(self.pf_dots, pf_idx, pf_sgts)

    def save_gct_points(self, prefix=""):
        """ Saves the coordinates of the Granule cell T points, i.e. the points where the Granule cell ascending axons
        split into the parallel fiber"""
        prefix = Path(prefix)
        assert hasattr(self, "aa_dots"), "No ascending axons added yet"
        gctp = self.aa_dots[:, -1, :]
        filename = prefix / "GCTcoordinates.sorted.dat"
        with filename.open("w") as f_out:
            f_out.write("\n".join(map(str_l, gctp)))
        print("Successfully wrote {}.".format(filename))


class MLI_pop(Cell_pop):
    """MLI cell population. Generates point representations of axons and dendrites as well as Query_point objects from them"""

    def __init__(self, my_args):
        Cell_pop.__init__(self, my_args)

    def add_axon(self):
        raise NotImplementedError("This part is not implemented yet.")

    def save_axon_coords(self, prefix=""):
        """ Save the coordinates of the dendrites, BREP style
        -> each line of the output file corresponds to one cell, and contains all its dendrites sequentially"""

        raise NotImplementedError("This part is not implemented yet.")

        # assert hasattr(self, "axon"), "Could not find axons, please generate them first"

        # prefix = Path(prefix)
        # axon_file = prefix / "MLIaxoncoordinates_test.dat"
        # with axon_file.open("w") as f_out:
        #     for ax in self.axon:
        #         flad = np.array([a for l in ax for a in l])
        #         f_out.write(str_l(flad) + "\n")
        #     print("Successfully wrote {}.".format(axon_file))

    def add_dendrites(self):
        coords, idx, segs = self.gen_dendrite(self.som)
        all_dends = coords
        all_idx = idx
        all_sgts = segs
        self.dends = Query_point(all_dends, all_idx, all_sgts)

    def save_dend_coords(self, prefix=""):
        """ Save the coordinates of the dendrites, BREP style
        -> each line of the output file corresponds to one cell, and contains all its dendrites sequentially"""

        raise NotImplementedError("This part is not implemented yet.")

        # assert hasattr(self, "dends"), "Could not find any added dendrites")

        # prefix = Path(prefix)

        # dend_file = prefix / "MLIdendcoordinates.dat"
        # with dend_file.open("w") as f_out:
        #     for ad in self.dend:
        #         flad = np.array([a for l in ad for a in l])
        #         f_out.write(str_l(flad) + "\n")
        # print("Successfully wrote {}.".format(dend_file))

    # TODO: input somas is only temporary and should replaced by self.som
    def gen_dendrite(self, return_end_points=False):

        somas = self.som*1.0
        MLzbegin = self.args.GLdepth + self.args.PCLdepth
        somas[:, 2] -= MLzbegin ## Shift from the top of PCL and GL layer to zero (z axis)

        Ldend = self.args.MLI_dend_length
 
        # generate end points in 2D
        def make_random_dend_ends():
            """ generates random end points in 2D. """

            angle0 = np.random.rand()*np.pi/2

            # first 3 angles between dends
            angles3 = np.ones(3)*np.pi/2
            angles = np.cumsum(angles3) 
            angles = np.hstack([0, angles]) + angle0 

            # adding more variability for each point independent from the others
            r=np.random.uniform(-0.5, 0.5,4)
            angles=angles+r

            # produce x and y coordinates
            return np.array([[Ldend*np.cos(a), Ldend*np.sin(a)] for a in angles]) # cos=x; sin=y

        def fix_dend_ends_upper(point1_xz, soma, upperLimit=self.args.MLdepth, push_down=100):
            """fixes end points lying beyond the boundaries."""
            radius = self.args.MLI_dend_length

            d=abs(soma[1]-upperLimit)
            
            point1_xz_new = np.zeros_like(point1_xz)
            for i in range(point1_xz.shape[0]):
                x, y=point1_xz[i];
                
                #print(point1_xz[i]);
                if y>soma[1]+d:

                    if point1_xz[i,0]< soma[0]:       # make sure each dendrite goes to one one side
                        y=soma[1]+d- np.random.uniform(0, push_down); #soma y coord. plus the distance to layer+ variability 
                        point1_xz_new[i,1]=y
                        phi=np.arcsin((y-soma[1])/radius) # calculate phi for new x
                        x1=radius*np.cos(phi)+soma[0]     # new x coord. on layer
                        point1_xz_new[i,0]=soma[0]-(x1-soma[0]) 
                    else:
                        y=soma[1]+d- np.random.uniform(0, push_down); #soma y coord. plus the distance to layer
                        phi=np.arcsin((y-soma[1])/radius) # calculate phi for new x
                        x1=radius*np.cos(phi)+soma[0]     # new x coord. on layer
                        point1_xz_new[i,:]=[x1, y]
                else:
                    point1_xz_new[i,:]= [x, y]

            return point1_xz_new

        def gen_dend_endpoints_2d(somas):
            """generates dendritic end points for somas."""
            # z-coordinates will lie between 0 to MLdepth
            upperLimit = self.args.MLdepth
            lowerLimit = 0
            
            EpointDend=np.empty((0,2), dtype=object) #empty array for dendrites

            for s in range(somas.shape[0]):
                soma = somas[s,:2]
                point1_xz = make_random_dend_ends() + soma
                point1_xz_new = fix_dend_ends_upper(point1_xz, soma, upperLimit)
                point1_xz_new = -fix_dend_ends_upper(-point1_xz_new, -soma, lowerLimit)
                EpointDend = np.append(EpointDend, point1_xz_new, axis=0)

            # print('EpointDend shape = ', EpointDend.shape)
            return EpointDend
        
        # Add a missing 3D coordinate to 2d end points
        def make_endpoint_3d(EpointDend, somas):
            ep1_3d_All=np.empty((0,3), dtype=object) #empty array for dendrites
            endpt_ids = np.empty(0, dtype=int) #empty array for dendrites

            sigma = self.args.MLI_x_sigma
            
            for s in range(somas.shape[0]):
                soma = somas[s,:]

                ep1 = EpointDend[(s*4):((s+1)*4),:] - somas[s,:2]
                ep1_3d = np.zeros((4,3))
                ep1_3d[:,:2] = ep1
                ep1_3d[:, 2] = np.random.randn(4)*sigma

                #print(ep1_3d)

                for i in range(4):
                    norm = np.sqrt(ep1_3d[i,0]**2 + ep1_3d[i,1]**2 + ep1_3d[i,2]**2)
                    ep1_3d[i,:] = ep1_3d[i,:]/norm*Ldend

                ep1_3d = ep1_3d + somas[s,:]
                ep1_3d_All=np.append(ep1_3d_All, ep1_3d, axis=0)
                    #print(EpointDend[(s*4):((s+1)*4),:])
                endpt_ids = np.concatenate((endpt_ids, np.zeros(4, dtype=int)+s))

            return ep1_3d_All, endpt_ids

        EpointDend = gen_dend_endpoints_2d(somas)
        ep1_3d_All, endpt_ids = make_endpoint_3d(EpointDend, somas)

        def generate_DendPoint_3d(soma, ep1_3d, endpt_ids, nPoint=90):
            """Generate Dendrititc Points between Soma and respective Endpoints
                In total 90 points are generated"""

            DendPointAll= np.empty((0,3),dtype=object)
            dendpt_ids = np.empty(0,dtype=int)
            segs = np.empty((0,2),dtype=int)

            for i in range(ep1_3d.shape[0]):
                VectorSoEp= ep1_3d[i]-soma #calculate the vector from soma to endpoint
                for s in range(1, nPoint+1):
                    DendPoint= np.array(soma + VectorSoEp/nPoint*s) # vector soma + vectorsomaendpoint shortened
                    DendPointAll = np.vstack((DendPointAll,DendPoint))
                    segs = np.vstack((segs, [i+1, int((s-1)/9)]))

                dendpt_ids = np.concatenate((dendpt_ids, np.zeros(nPoint, dtype=int)+endpt_ids[i]))

            return DendPointAll, dendpt_ids, segs
        
        ##generate endpoints around respective somas
        DendPointAllAll = np.empty((0,3), dtype=object) #empty array for dendrites
        dendpt_ids_all = np.empty((0,1), dtype=int) #empty array for dendrite ids
        segs_all = np.empty((0,2), dtype=int) #empty array for dendrite ids

        for s in trange(somas.shape[0]):
            soma = somas[s]
            DendPointAll, dendpt_ids, segs = generate_DendPoint_3d(soma, ep1_3d_All[s*4:(s+1)*4], endpt_ids[s*4:(s+1)*4])
    
            DendPointAllAll = np.vstack((DendPointAllAll, DendPointAll))
            dendpt_ids_all = np.append(dendpt_ids_all, dendpt_ids)
            segs_all = np.vstack((segs_all, segs))

        DendPointAllAll[:, 2] += MLzbegin # Put every point above the PCL

        if return_end_points:
            return DendPointAllAll, dendpt_ids_all, segs_all, ep1_3d_All
        else:
            return DendPointAllAll, dendpt_ids_all, segs_all



# class Map(dict):
#     """
#     Example:
#     m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
#     """
#     def __init__(self, *args, **kwargs):
#         super(Map, self).__init__(*args, **kwargs)
#         for arg in args:
#             if isinstance(arg, dict):
#                 for k, v in iter(arg.items()):
#                     self[k] = v
#
#         if kwargs:
#             for k, v in iter(kwargs.items()):
#                 self[k] = v
#
#     def __getattr__(self, attr):
#         return self.get(attr)
#
#     def __setattr__(self, key, value):
#         self.__setitem__(key, value)
#
#     def __setitem__(self, key, value):
#         super(Map, self).__setitem__(key, value)
#         self.__dict__.update({key: value})
#
#     def __delattr__(self, item):
#         self.__delitem__(item)
#
#     def __delitem__(self, key):
#         super(Map, self).__delitem__(key)
#         del self.__dict__[key]
#
if __name__ == "__main__":
    # TODO: old script for test runs
    a = Cell_pop(3)
    som_coord = a.gen_random_cell_loc(10, 1500, 750, 200, 2)
    print(som_coord)
    mlis = MLI_pop([])
    print(mlis)
    # print(a.load_somata(som_coord))
    # print(mlis.add_dendrites())
    print(mlis.gen_dendrite())
