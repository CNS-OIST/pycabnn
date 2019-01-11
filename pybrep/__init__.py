"""
BREPpy.py

Finds distance-based connectivity between neurons with spatially extended
dendritic and axonal morphology, mainly developed for a physiologically detailed
network model of the cerebellar cortex.

Written by Ines Wichert and Sungho Hong, Computational Neuroscience Unit,
Okinawa Institute of Science and Technology

"""
import numpy as np
import datetime
import csv
import warnings

from tqdm import tqdm
from pathlib import Path
from sklearn.neighbors import KDTree

####################################################################
## GENERAL UTILS PART                                                 ##
####################################################################
from util import str_l

####################################################################
## CONNECTOR PART                                                 ##
####################################################################

class Connect_3D(object):
    ''' Connects 3D point datasets.'''

    def __init__(self, qpts_src, qpts_tar, c_rad, prefix=''):
        '''Connect two 3D point datasets with a given critical radius.
        qpts_src: source population
        qpts_tar: target population
        c_rad: critical radius, points from distinct datasets that have a distance of less than c_rad will be connected
        prefix: For the saving procedure'''

        #If the arguments are np arrays, make a Query_point from it.
        if type(qpts_src).__module__ == np.__name__ : qpts_src = Query_point(qpts_src)
        if type(qpts_tar).__module__ == np.__name__ : qpts_tar = Query_point(qpts_tar)
        #  save to object
        self.spts = qpts_src
        self.tpts = qpts_tar
        self.c_rad = c_rad
        self.prefix = prefix

    def connections_parallel(self, deparallelize=False, src_in_tree=[]):
        ''' Finds the connections in parallel, depending on a running IPython cluster.
        The deparallelize option switches to a serial mode independent of the Ipython cluster.
        src_in_tree determines whether the source or target population will go into the tree.
        If ĺeft blank, the bigger point cloud will be used, as this is faster.'''

        if not deparallelize:
            # set up client and workers
            import ipyparallel as ipp
            self.rc = ipp.Client()
            self.dv = self.rc[:]
            self.dv.use_cloudpickle() #this is necessary so that closures can be sent to the workers
            self.lv = self.rc.load_balanced_view()
            import os
            print('Started parallel process with', len(self.rc.ids), 'workers.')
            print('Work directory for workers:', self.rc[0].apply_sync(os.getcwd))
            print('Work directory of main process:', os.getcwd())

        # If not given, determine which point cloud should go to the tree (the bigger one)
        if not type(src_in_tree).__name__ == 'bool':
            src_in_tree = self.spts.npts > self.tpts.npts
        self.src_in_tree = src_in_tree

        #depending on which points shall be in the tree and which shall be the query points, assign, and project at the same time
        if src_in_tree:
            kdt = KDTree(self.spts.coo)
        else:
            kdt = KDTree(self.tpts.coo)

        #import methods to workers
        if not deparallelize:
            self.dv.block = True
            with self.dv.sync_imports():
                import parallel_util

        # Using cloudpickle, the variables used in the worker methods have to be in the workspace
        spts = self.spts
        tpts = self.tpts
        c_rad = self.c_rad
        src_in_tree = self.src_in_tree
        prefix = self.prefix

        print('sit', src_in_tree)

        lam_qpt = lambda ids: parallel_util.find_connections_3dpar(kdt, spts, tpts, c_rad, src_in_tree, ids, prefix)

        if src_in_tree: nqpts = tpts.npts
        else: nqpts = spts.npts

        if not deparallelize:
            #chunk the query points so that each worker gets roughly the same amount of points to query in the KDTree
            #Only IDs are sent in order to speed it up by preallocating the coordinates and as much information as possible
            #to the workers.
            id_sp = int(np.ceil(nqpts/len(self.rc.ids)))
            id_ar = [np.arange(id_sp) + i * id_sp for i in range(int(np.ceil(nqpts/id_sp)))]
            id_ar[-1] = id_ar[-1][id_ar[-1]<nqpts]
            id_ar= [[i, id_ar[i]] for i in range(len(id_ar))]
            s = list(self.lv.map(lam_qpt, id_ar, block = True))
        else:
            import parallel_util
            s = lam_qpt([0, np.arange(nqpts)])


        print('Exited process, saving as:' , prefix)


class Connect_2D(object):
    '''Finds connections between one structure that can be projected to a 2D plane and another structure
    that is represented by 3D coordinate points.'''

    def __init__(self, qpts_src, qpts_tar, c_rad, prefix='', table_name='connection', save_mode='sqlite'):
        '''Initialize the Connect_2D object. Source population, target population, critical radius.
        The source and target population can either be of the Query_point class or arrays'''

        #If the arguments are np arrays, make a Query_point from it.
        if type(qpts_src).__module__ == np.__name__:
            qpts_src = Query_point(qpts_src)
        if type(qpts_tar).__module__ == np.__name__:
            qpts_tar = Query_point(qpts_tar)

        #find out which of the pointsets is the linear one and assign accordingly
        if qpts_src.lin and not qpts_tar.lin:
            self.lpts = qpts_src
            self.pts = qpts_tar
            self.lin_axis = qpts_src.lin_axis
        elif qpts_tar.lin and not qpts_src.lin:
            self.lpts = qpts_tar
            self.pts = qpts_src
            self.lin_axis = qpts_tar.lin_axis
        else:
            print('Failure to initialize connector, there is not one linearized and one regular point set')
        self.lin_is_src = qpts_src.lin
        self.c_rad = c_rad
        self.save_mode = save_mode
        self.prefix = Path(prefix)
        self.table_name = table_name


    def connections_parallel(self, deparallelize=False, serial_fallback=False, req_lin_in_tree=[], nblocks=None, run_only=[], debug=False):
        '''searches connections, per default in parallel. Workers will get a copy of the tree, query points etc.
        and perform the search independently, each saving the result themself.
        deparallelize: if set True, modules and functions tailored for a parallel process will be used,
        but all will be run in serial (no iPython cluster necessary)
        serial_fallback:  If set True, this function will call the older connect function
        nblocks: Number of blocks that the data should be partitioned into'''

        # If not deparallelized, try to connect to the parallel cluster, fall back to serial if necessary
        if not deparallelize:
            try:
                import ipyparallel as ipp
                self.rc = ipp.Client()
                self.dv = self.rc[:]
                self.dv.use_cloudpickle()
                self.lv = self.rc.load_balanced_view()
                if nblocks is None:
                    nblocks = len(self.rc.ids)
            except:
                print('Parallel process could not be started!')
                if serial_fallback:
                    print('Will do it sequentially instead...')
                    res, l_res = self.find_connections()
                    self.save_results(res, l_res, self.prefix)
                return
        else:
            if nblocks is None:
                nblocks = 1

        print('Blocks = ', nblocks)

        # Get tree setup
        if type(req_lin_in_tree).__name__ == 'bool':
            kdt, q_pts = self.get_tree_setup(return_lax=False, lin_in_tree=req_lin_in_tree)
        else:
            kdt, q_pts = self.get_tree_setup(return_lax=False)

        # Copy variables to workspace for cloudpickle
        pts = self.pts
        lpts = self.lpts
        c_rad = self.c_rad
        lin_axis = self.lin_axis
        lin_in_tree = self.lin_in_tree
        lin_is_src = self.lin_is_src
        save_mode = self.save_mode
        prefix = str(self.prefix)
        table_name = self.table_name

        if save_mode=='sqlite':
            import sqlite3
            conn = sqlite3.connect(prefix+'.db')
            c = conn.cursor()
            warnings.warn('Pre-existing table ' + table_name + ' will be destroyed.')
            c.execute('DROP TABLE IF EXISTS ' + table_name)
            conn.commit()
            conn.close()

        lam_qpt = lambda ids: parallel_util.find_connections_2dpar(kdt, pts, lpts, c_rad, lin_axis, lin_in_tree, lin_is_src, ids, prefix, table_name, save_mode, debug)

        # split data into nblocks blocks
        n_q_pts = len(q_pts)
        id_ar = np.array_split(np.arange(n_q_pts), nblocks)
        if run_only:
            id_ar = [(i, id_ar[i]) for i in run_only]
        else:
            id_ar = [(i, id_ar[i]) for i in range(nblocks)]
        # print(id_ar) # Check what is in id_ar

        if not deparallelize:
            with self.dv.sync_imports():
                import parallel_util
            s = list(self.lv.map(lam_qpt, id_ar, block=True))
        else:
            # essentially the same as connections_pseudo_parallel
            import parallel_util
            s = []
            for id1 in tqdm(id_ar):
                # print('Processing block:', id1[0])
                #print('Points:', id1[1])
                s.append(lam_qpt(id1))

        print('Exited process, saving as: {}.'.format(prefix))
        return s


    def get_tree_setup(self, return_lax=True, lin_in_tree=[]):
        '''Gets the setup for the connection.
        lin_in_tree determines whether
         '''

        # if not specified which of the point population is the query point population, take the smaller one and put the bigger one in the tree.
        if not type(lin_in_tree).__name__ == 'bool':
            lin_in_tree = self.lpts.npts > self.pts.npts
        self.lin_in_tree = lin_in_tree
        #depending on which points shall be in the tree and which shall be the query points, assign, and project at the same time
        if lin_in_tree:
            tr_pts = self.lpts.coo[:, 0, np.invert(self.lin_axis)]
            q_pts = self.pts.coo[:, np.invert(self.lin_axis)]
        else:
            q_pts = self.lpts.coo[:, 0, np.invert(self.lin_axis)]
            tr_pts = self.pts.coo[:, np.invert(self.lin_axis)]

        #build 2D Tree
        kdt = KDTree(tr_pts)

        if not return_lax:
            return kdt, q_pts
        else:
            #get the information for the axis along which the projection is done
            lax_c = self.pts.coo[:,self.lin_axis]
            lax_range = self.lpts.coo[:,:,self.lin_axis]
            lax_range = lax_range.reshape((lax_range.shape[0], lax_range.shape[1]))
            return kdt, q_pts, lax_c, lax_range, self.lin_in_tree

    def find_connections(self, lin_in_tree=[]):
        '''Perform the seperate nearest-neighbour searches in the tree'''

        kdt, q_pts, lax_c, lax_range, _ = self.get_tree_setup(True, lin_in_tree)
        res = []
        l_res = []
        for i, pt in enumerate(q_pts): #iterate through the query points
            # find the points within the critical radius
            warnings.simplefilter('ignore')
            ind, = kdt.query_radius(np.expand_dims(pt, axis = 0), r = self.c_rad)
            #check if the found points match along the linearized axis and if so, add distance from the beginning of the linearized axis
            if self.lin_in_tree:
                ind = ind[np.logical_and(lax_range[ind,0]<=lax_c[i], lax_range[ind,1]>= lax_c[i])]
                l_res.append(abs(lax_c[i] - lax_range[ind,0] - self.lpts.set_0[ind])+ self.lpts.lin_offset[ind])
            else:
                ind = ind[np.logical_and(lax_range[i,0]<=lax_c[ind], lax_range[i,1]>= lax_c[ind]).ravel()]
                l_res.append(abs(lax_c[ind].ravel() - lax_range[i,0] -self.lpts.set_0[i]) + self.lpts.lin_offset[i])

            res.append(ind.astype('int'))

        return res, l_res


    def save_results(self, res, res_l, prefix=''):
        '''Saves the results as produced by the query_x_in_y method similarly as BREP.
        res = result(containing a list of arrays/lists with the found IDs
            -> first index = query point ID
            -> numbers within the 2nd level arrays/lists -> tree point IDs
        res_l = for the linear point species, distance from the first outer boder
        prefix = prefix for the result file -> path, output specifier
        With the two last bools different 4 different modes can be created to have flexibility.'''

        prefix = str(prefix)
        # file names
        fn_tar = prefix + 'targets.dat'
        fn_src = prefix + 'sources.dat'
        fn_segs = prefix + 'segments.dat'
        fn_dis = prefix + 'distances.dat'

        with open(fn_tar, 'w') as f_tar, open(fn_src, 'w') as f_src, open(fn_dis, 'w') as f_dis, open(fn_segs, 'w') as f_segs:

            for l,(cl, cl_l) in enumerate(zip(res, res_l)):

                assert len(cl) == len(cl_l), 'Something went wrong, all corresponding lists in your input arguments should have the same length'
                assert hasattr(self, 'lin_in_tree'), 'Attribute lin_in_tree must be set, this should happen in the get_tree_setup method!'
                assert hasattr(self, 'lin_is_src'), 'Attribute lin_is_src must be set, this should happen at initialization depending on the order of the query point arguments!'

                if len(cl_l)>0:
                    f_dis.write("\n".join(map(str, cl_l)))
                    #first, get the cell IDS of the query and tree points(for the linear points, that is just the point ID,
                    #for the other points this information has to be extracted from the corresponding Query_points object.
                    #Segments also corresponds to the 3D point population, right value is acquired from Query-points object.
                    if self.lin_in_tree:
                        s_ar = self.pts.seg[l,:].astype('int')
                        f_segs.write("\n".join(map(str_l, [s_ar for i in range(len(cl))])))#*np.ones((len(cl), len(s_ar))))))

                        q_id = (np.ones(len(cl))*self.pts.idx[l]).astype('int')
                        tr_id = cl
                    else:
                        f_segs.write("\n".join(map(str_l, self.pts.seg[cl].astype('int'))))
                        q_id = self.pts.idx[cl].ravel()
                        tr_id = (np.ones(len(cl))*l).astype('int')

                    #depending on which population should be source and which should be target, save cell IDs accordingly.
                    if self.lin_is_src:
                        f_src.write("\n".join(map(str, tr_id)))
                        f_tar.write("\n".join(map(str, q_id)))
                    else:
                        f_src.write("\n".join(map(str, q_id)))
                        f_tar.write("\n".join(map(str, tr_id )))

                    #need to attach one more line here or we get two elements per line
                    f_dis.write("\n")
                    f_src.write("\n")
                    f_tar.write("\n")
                    f_segs.write("\n")


class Query_point(object):
    def __init__(self, coord, IDs = None, segs = None, lin_offset = 0, set_0 = 0, prevent_lin = False):
        '''Make a Query_point object from a point array and any meta data:
        The coord array should have either the shape (#points, point dimension) or
        (#cells, #points per cell, point dimenstion). In the second case the array will be reshaped to be like the first case,
        with the additional attributes IDs (cell, first dimenion of coord), and segs (second dimension of coord).
        It will be automatically checked whether the points can be linearized/projected, i.e. represented by a start, end, and 2-D projection'''

        self.npts = len(coord)
        # check if lin -> then it can be used for the Connect_2D method. In that case it will not be
        if not prevent_lin:
            self.lin = self.lin_check(coord)
            if self.lin:
                #lin_offset will be added to the distance for each connection (e.g. aa length for pf)
                try:
                    lin_offset = float(np.array(lin_offset)) * np.ones(self.npts)
                except:
                    assert(len(lin_offset) == self.npts), 'lin_offset should be a scalar or an array with length npts!'
                finally:
                    self.lin_offset = lin_offset
                #set0 sets where 0 is defined along the elongated structure (e.g. branching point for PF)
                try:
                    set_0 = float(np.array(set_0)) * np.ones(self.npts)
                except:
                    assert(len(set_0) == self.npts), 'lin_offset should be a scalar or an array with length npts!'
                finally: self.set_0 = set_0

                self.coo = coord
                self.seg = np.ones(len(coord))
                if IDs is None:
                    self.idx = np.arange(len(coord))
                else:
                    assert len(IDs) == len(coord), 'ID length does not match '
                    self.idx = IDs
                return
        self.lin = False


        # If the structure is already flattened or there is only one point per ID
        if len (coord.shape) == 2:
            self.coo = coord
            if IDs is not None:
                assert len(coord) == len(IDs), 'Length of ID list and length of coordinate file must be equal'
                self.idx = IDs
            if segs is None:
                self.seg = np.ones(len(coord))
                if IDs is None:
                    self.idx = np.arange(len(coord))
            else:
                assert not np.all(IDs == None), 'Cell IDs must be sepcified before segment number'
                self.seg = segs

        # If the input array still represents the structure
        if len(coord.shape) == 3:
            assert (IDs is None) == (segs is None), 'To avoid confusion, cell IDs and segment numbers must either be specified both, or neither'
            if (not (IDs is None)) and (not (segs is None)):
                assert np.all(IDs.shape == coord.shape[:-1]) and np.all(segs.shape == coord.shape[:-1]) , 'Dimensions of ID and segment file should be '+str(coord.shape[:-1])
            else:
                IDs = np.array([[[i] for j in range(coord.shape[1])] for i in range(coord.shape[0])])
                segs = np.array([[[j] for j in range(coord.shape[1])] for i in range(coord.shape[0])])
            lam_res = lambda d: d.reshape(d.shape[0]*d.shape[1],d.shape[2])
            self.coo = lam_res(coord)
            self.seg = lam_res(np.expand_dims(segs, axis = 2))
            self.idx = lam_res(np.expand_dims(IDs, axis = 2))


    #check if input array can be used for the projection method (Connect_2D) (-> the points have a start and an end point)
    #Note that a lot of points describing a line will not be projected automatically
    def lin_check(self, coord):
        if len(coord.shape) == 3:
            if coord.shape[1] == 2 and coord.shape[2] == 3:
                sm = sum(abs(coord[:,0,:] - coord[:,1,:]))
                no_dif = [np.isclose(sm[i],0) for i in range(len(sm))]
                if sum(no_dif) == 2: # 2 coordinates are the same, one is not
                    self.lin_axis = np.invert(no_dif) #this one is the axis that cn be linearized
                    return True
        return False


    def linearize(self):
        pass
        #this function should linearize points when they are in a higher structure than nx3, and the IDs and


####################################################################
## POPULATION PART                                                ##
####################################################################

class Cell_pop(object):

    def __init__(self, my_args):
        self.args = my_args

    def load_somata(self, fn_or_ar):
        ''' Loads somata and adds them to the population object:
        Either from a coordinate file(if fn_or_ar is a filename)
        Or diectly from a numpy array(if fn_or_ar is such)'''
        if hasattr(fn_or_ar, 'shape'):
            try:
                if fn_or_ar.shape[len(fn_or_ar.shape)-1] == 3:
                    self.som = fn_or_ar
                else:
                    print('Cannot recognize array as coordinates')
            except:
                print('Could not read in soma points')
        else:
            try:
                self.read_in_soma_file(fn_or_ar)
                print('Successfully read {}.'.format(fn_or_ar))
            except:
                print('Tried to read in ', fn_or_ar, ', failed.')

    def save_somata(self, prefix='', fn=''):
        '''Save the soma coordinates'''
        prefix = Path(prefix)
        if fn == '': fn = type(self).__name__ + '_coords.dat'
        '''Save the soma coordinates'''
        assert hasattr(self, 'som'), 'Cannot save soma coordinates, as apparently none have been added yet'
        with (prefix / fn).open('w') as f_out:
            f_out.write("\n".join(map(str_l, self.som)))
        print('Successfully wrote {}.'.format(prefix / fn))

    def read_in_soma_file(self, fn, parse_ignore = True):
        ''' Reads in files such as the ones that BREP returns.
        Represents lines as rows, nth element in each line as column.
        fn = Filename
        parse_ignore: If something cannot be parsed, it will be ignored. If this parameter is set false, it will complain
        returns: 2d-array of floats'''
        res = []
        with open(fn, 'r', newline = '') as f:
            rr = csv.reader(f, delimiter = ' ')
            err = [] # list of elements that could not be read in
            for line in rr: # lines -> rows
                ar = []
                for j in range(len(line)):
                    try: ar.append(float(line[j]))
                    except: err.append(line[j])
                res.append(np.asarray(ar))
        if len(err)> 0 and not parse_ignore: print('Could not parse on {} instances: {}'.format(len(err), set(err)))
        self.som = np.asarray(res)
        self.n_cell = len(self.som)


    def coord_reshape(dat, n_dim = 3):
        ''' Reshapes coordinate files with several points in one line by adding an extra axis.
        Thus, converts from an array with shape(#cells x(#pts*ndim)) to one with shape(#cells x #pts x ndim)'''
        dat = dat.reshape([dat.shape[0], int(dat.shape[1]/n_dim),n_dim])
        return dat


    def gen_random_cell_loc(self, n_cell, Gc_x = -1, Gc_y = -1, Gc_z = -1, sp_std = 2):
        '''Random generation for cell somatas
        n_cell(int) = number of cells
        Gc_x, Gc_y, Gc_z(float) = dimensions of volume in which cells shall be distributed
        Algorithm will first make a grid that definitely has more elements than n_cell
        Each grid field is populated by a cell, then those cells are displaced randomly
        Last step is to prune the volume, i.e. remove the most outlying cells until the goal number of cells is left
        Returns: cell coordinates'''
        # get spacing for grid:
        if Gc_x < 0: Gc_x = self.args.GoCxrange
        if Gc_y < 0: Gc_y = self.args.GoCyrange
        if Gc_z < 0: Gc_z = self.args.GoCzrange

        vol_c = Gc_x*Gc_y*Gc_z/n_cell #volume that on average contains exactly one cell
        sp_def = vol_c**(1/3)/2 #average spacing between cells(cube of above volume)

        #Get grid with a few too many elements
        gr = np.asarray([[i,j,k]
                         for i in np.arange(0, Gc_x, 2*sp_def)
                         for j in np.arange(0, Gc_y, 2*sp_def)
                         for k in np.arange(0, Gc_z, 2*sp_def)])
        #random displacement
        grc = gr + np.random.randn(*gr.shape)*sp_def*sp_std

        #then remove the ones that lie most outside to get the correct number of cells:
        #First find the strongest outliers
        lower = grc.T.ravel()
        upper = -(grc-[Gc_x, Gc_y, Gc_z]).T.ravel()
        most_out_idx = np.mod(np.argsort(np.concatenate((lower,upper))), len(grc))
        #In order to find the right number, must iterate a bit as IDs may occur twice(edges)
        del_el = len(grc) - n_cell # number of elements to be deleted
        n_del = del_el
        while len(np.unique(most_out_idx[:n_del])) < del_el:
            n_del = n_del + del_el - len(np.unique(most_out_idx[:n_del]))
        #Deletion step
        grc = grc[np.setdiff1d(np.arange(len(grc)), most_out_idx[:n_del]),:]

        #Now, this might still bee too far out, so we shrink or expand it.
        mi = np.min(grc, axis = 0)
        ma = np.max(grc, axis = 0)
        dim  =([Gc_x, Gc_y, Gc_z])
        grc =(grc - mi)/(ma-mi)*dim

        self.som = grc
        return grc


class Golgi_pop(Cell_pop):
    '''Golgi cell population. Generates point representations of axons and dendrites as well as Query_point objects from them'''

    def __init__(self, my_args):
        Cell_pop.__init__(self,my_args)

    def add_axon (self):
        '''Adds axons as points with a uniform random distribution in a certain rectangle
        The seg array will contain the distance of the axon point from the soma'''
        x_r = [self.args.GoC_Axon_Xmin, self.args.GoC_Axon_Xmax]
        y_r = [self.args.GoC_Axon_Ymin, self.args.GoC_Axon_Ymax]
        z_r = [self.args.GoC_Axon_Zmin, self.args.GoC_Axon_Zmax]
        n_ax = int(self.args.numAxonGolgi)

        ar = np.random.uniform(size = [len(self.som), n_ax+1, 3])
        for i, [low, high] in enumerate([x_r, y_r, z_r]):
            ar[:,:,i] = ar[:,:,i]*(high-low)+low
        ar[:,0,:] = ar[:,0,:]*0
        for i in range(len(ar)):
            ar[i,:,:] = ar[i,:,:] + self.som[i,:]
        segs = np.linalg.norm(ar, axis = 2)
        idx = np.array([[j for k in range(len(ar[j]))] for j in range(len(ar))])
        self.axon = ar
        self.axon_q = Query_point(ar, idx, segs)

    def save_axon_coords(self, prefix=''):
        ''' Save the coordinates of the dendrites, BREP style
        -> each line of the output file corresponds to one cell, and contains all its dendrites sequentially'''
        assert hasattr(self, 'axon'), 'Could not find axon, please generate first'

        prefix = Path(prefix)
        axon_file = prefix /  'GoCaxoncoordinates_test.dat'
        with axon_file.open( 'w') as f_out:
            for ax in self.axon:
                flad = np.array([a for l in ax for a in l])
                f_out.write(str_l(flad)+"\n")
            print('Successfully wrote {}.'.format(axon_file))


    def add_dendrites(self):
        '''Add apical and basolateral dendrites using the parameters specified in the Parameters file.
        Will construct Query points by itself and add them to the object'''
        #apical
        a_rad = self.args.GoC_PhysApicalDendR #radius of cylinder
        a_h = self.args.GoC_PhysApicalDendH   #height of cylinder
        a_ang = [self.args.GoC_Atheta_min, self.args.GoC_Atheta_max] # mean angles for dendrite direction
        a_std = self.args.GoC_Atheta_stdev  # std for dendrite diredtion
        a_n = int(self.args.GoC_Ad_nseg * self.args.GoC_Ad_nsegpts) # number of points per dendrite

        #basolateral
        b_rad = self.args.GoC_PhysBasolateralDendR
        b_h = self.args.GoC_PhysBasolateralDendH
        b_ang = [self.args.GoC_Btheta_min, self.args.GoC_Btheta_max]
        b_std = self.args.GoC_Btheta_stdev
        b_n = int(self.args.GoC_Bd_nseg * self.args.GoC_Bd_nsegpts)

        #generate the dendrite coordinates
        a_dend, a_idx, a_sgts = self.gen_dendrite(a_rad, a_h, a_ang, a_std, a_n)
        b_dend, b_idx, b_sgts = self.gen_dendrite(b_rad, b_h, b_ang, b_std, b_n)

        #The apical dendrites have higher numbers than the basal ones:
        a_sgts[:,:,1] = a_sgts[:,:,1] + len(b_ang)
        #Taking into account that there are several points per segment and the first segment has index 1
        a_sgts[:,:,0] = np.floor(a_sgts[:,:,0]/self.args.GoC_Ad_nsegpts)+1
        b_sgts[:,:,0] = np.floor(b_sgts[:,:,0]/self.args.GoC_Bd_nsegpts)+1

        # special concatenation function for apical and basal dendrite
        # stack coords/segments of a and b dends when they are from the same cell
        conc_ab_one = lambda i, a, b: np.vstack((a[a_idx==i], b[b_idx==i]))
        # loop around all the cells
        conc_ab = lambda a, b: np.vstack(conc_ab_one(i, a, b) for i in range(self.n_cell))

        #concatenated dendrite information(coords, cell indices, segment information)
        all_dends = conc_ab(a_dend, b_dend)
        all_sgts  = conc_ab(a_sgts, b_sgts)

        # put a and b indices together and rearrange them in a row
        all_idx = np.hstack((a_idx, b_idx))
        all_idx = all_idx.reshape((np.prod(all_idx.shape),1))

        # test code for the part above
        # zz = all_dends[all_idx.flatten()==2,:]
        # plt.plot(zz[:,1], zz[:,2],'.')

        self.a_dend = a_dend
        self.b_dend = b_dend
        self.qpts = Query_point(all_dends, all_idx, all_sgts)



    def save_dend_coords(self, prefix=''):
        ''' Save the coordinates of the dendrites, BREP style
        -> each line of the output file corresponds to one cell, and contains all its dendrites sequentially'''
        assert hasattr(self, 'a_dend') or hasattr(self, 'b_dend'), 'Could not find any added dendrites'

        prefix = Path(prefix)

        if hasattr(self, 'a_dend'):
            dend_file = prefix /  'GoCadendcoordinates.dat'
            with dend_file.open( 'w') as f_out:
                for ad in self.a_dend:
                    flad = np.array([a for l in ad for a in l])
                    f_out.write(str_l(flad)+"\n")
            print('Successfully wrote {}.'.format(dend_file))
        else:
            warnings.warn('Could not find apical dendrite')

        if hasattr(self, 'b_dend'):
            dend_file = prefix /  'GoCbdendcoordinates.dat'
            with dend_file.open('w') as f_out:
                for bd in self.b_dend:
                    flbd = [b for l in bd for b in l]
                    f_out.write(str_l(flbd)+"\n")
            print('Successfully wrote {}.'.format(dend_file))
        else:
            warnings.warn('Could not find basal dendrite')


    def gen_dendrite(self, c_r, c_h, c_m, c_std, c_n):
        '''Generates dendrites as described in the paper:
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
        '''
        c_gr = np.linspace(0,1,c_n)*np.ones((3, c_n)) #linspace grid between 0 and 1 with c_n elements
        b_res = []
        idx = []  #cell indices

        c_m = -np.array(c_m) + 90 # This is the angle conversion that is necessary to be compatible with the scheme BREP

        for i in range(len(self.som)): #each cell
            som_c = self.som[i,:]
            d_res = []
            if i == 0: d_sg = [] #segments, only have to be calculated once as they are the same for every cell
            for n,cc_m in enumerate(c_m): #each dendrite
                ep_ang =(np.random.randn()*c_std + cc_m)*np.pi/180 #individual angle
                pt =([np.sin(ep_ang)*c_r, np.cos(ep_ang)*c_r, c_h])*c_gr.T #coordinates of the dendrite = endpoint*grid
                d_res = d_res + list(pt+som_c)
                if i == 0: d_sg = d_sg + list(np.array([np.arange(c_n), np.ones(c_n)*(n+1)]).T)
            b_res.append(np.array(d_res))
            idx.append((np.ones(len(d_res))*i).astype('int'))
        segs = np.array([d_sg for k in range(i+1)]) #replicate segment information for each cell

        return np.array(b_res), np.array(idx), segs



class Granule_pop(Cell_pop):
    '''Granule cell population. Generates point representations of ascending axon (aa) and parallel fiber(pf)
    Can either do so with 3D points or with 2D projections. AA length can be fixed or random'''
    def __init__(self, my_args):
        Cell_pop.__init__(self, my_args)
        self.aa_length = self.args.PCLdepth+ self.args.GLdepth


    def add_aa_endpoints_random(self):
        '''Generate aa endpoints with a random aa length(end point will be somewhere in mol_range)'''
        mol_range = [self.aa_length, self.aa_length+self.args.MLdepth]
        self.aa_dots = np.array([np.array([self.som[i], self.som[i]]) for i in range(len(self.som))])
        self.aa_dots[:,1,2] = np.random.uniform(mol_range[0], mol_range[1], len(self.aa_dots[:,1,2]))
        self.qpts_aa = Query_point(self.aa_dots)


    def add_aa_endpoints_fixed(self):
        '''Generate aa endpoints with a fixed aa length'''
        #aa_length = self.args.PFzoffset   #NOTE: This value exists, but in the BREP original file it is replaced by the other definition
        self.aa_dots = np.array([np.array([self.som[i], self.som[i]]) for i in range(len(self.som))])
        self.aa_dots[:,1,2] = self.aa_dots[:,1,2] + self.aa_length
        self.qpts_aa = Query_point(self.aa_dots)


    def add_pf_endpoints(self):
        ''' Add the endpoints of the parallel fibers [begin, end] for each cell'''
        pf_length = self.args.PFlength
        assert hasattr(self, 'aa_dots'), 'Cannot add Parallel Fiber, add ascending axon first!'
        self.pf_dots = self.aa_dots.copy()
        self.pf_dots[:,0,2] = self.pf_dots[:,1,2] #z axis shall be the same
        self.pf_dots[:,0,0] = self.pf_dots[:,0,0] - pf_length/2
        self.pf_dots[:,1,0] = self.pf_dots[:,1,0] + pf_length/2
        self.qpts_pf = Query_point(self.pf_dots, lin_offset = self.aa_dots[:,1,2] - self.aa_dots[:,0,2], set_0 = pf_length/2)


    def add_3D_aa_and_pf(self):
        '''adds 3-dimensional coordinates for ascending axons and parallel fiber to the granule cell objects.
        Both AA and PF are represented by regularly spaced dots'''

        aa_length = self.args.PCLdepth+ self.args.GLdepth
        aa_nd = int(self.aa_length / self.args.AAstep) #number of dots for the aa
        aa_sp = np.linspace(0, self.aa_length, aa_nd) #grid that contains the spacing for the aa points

        pf_nd = int(self.args.PFlength/self.args.PFstep) # number of dots for the pf
        pf_sp = np.linspace(-self.args.PFlength/2, self.args.PFlength/2, pf_nd) # grid that contains spacing of po points

        self.aa_dots = np.zeros((len(self.som), aa_nd, 3))
        self.pf_dots = np.zeros((len(self.som), pf_nd, 3))
        aa_idx = np.zeros((len(self.som), aa_nd))
        aa_sgts= np.zeros((len(self.som), aa_nd))
        pf_idx = np.zeros((len(self.som), pf_nd))
        pf_sgts= np.zeros((len(self.som), pf_nd))

        for i, som in enumerate(self.som):

            self.aa_dots[i] = np.ones((aa_nd, 3))*som #copy soma location for each point of the aa
            self.aa_dots[i,:,2] = self.aa_dots[i,:,2] + aa_sp # add the z offsets
            aa_idx[i,:] = i #cell indices, for the query object
            aa_sgts[i,:] = np.arange(aa_nd) #segment points, for the query object

            self.pf_dots[i] = np.ones((pf_nd,3))*self.aa_dots[i,-1, :] #uppermost aa point is the origin of the pf points
            self.pf_dots[i,:,0] = self.pf_dots[i,:,0] + pf_sp #this time, points differ only by their offset along the x direction
            pf_idx[i,:]  = i
            pf_sgts[i,:] = np.arange(pf_nd) #! Not necessarily nice

        self.qpts_aa = Query_point(self.aa_dots, aa_idx, aa_sgts)
        self.qpts_pf = Query_point(self.pf_dots, pf_idx, pf_sgts)


    def save_gct_points(self, prefix=''):
        ''' Saves the coordinates of the Granule cell T points, i.e. the points where the Granule cell ascending axons
        split into the parallel fiber'''
        prefix = Path(prefix)
        assert hasattr(self, 'aa_dots'),  'No ascending axons added yet'
        gctp = self.aa_dots[:,-1,:]
        filename = prefix / 'GCTcoordinates.sorted.dat'
        with filename.open('w') as f_out:
            f_out.write("\n".join(map(str_l, gctp)))
        print('Successfully wrote {}.'.format(filename))
