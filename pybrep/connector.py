####################################################################
## CONNECTOR PART                                                 ##
####################################################################

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KDTree
from .util import str_l, Query_point
import warnings


class Connect_3D(object):
    ''' Connects 3D point datasets.'''

    def __init__(self, qpts_src, qpts_tar, c_rad, prefix=''):
        '''Connect two 3D point datasets with a given critical radius.
        qpts_src: source population
        qpts_tar: target population
        c_rad: critical radius, points from distinct datasets that have a distance of less than c_rad will be connected
        prefix: For the saving procedure'''

        # If the arguments are np arrays, make a Query_point from it.
        if type(qpts_src).__module__ == np.__name__:
            qpts_src = Query_point(qpts_src)
        if type(qpts_tar).__module__ == np.__name__:
            qpts_tar = Query_point(qpts_tar)
        #  save to object
        self.spts = qpts_src
        self.tpts = qpts_tar
        self.c_rad = c_rad
        self.prefix = prefix

    def connections_parallel(self, deparallelize=False, src_in_tree=[], use_distance='add', avoid_self=True):
        ''' Finds the connections in parallel, depending on a running IPython cluster.
        The deparallelize option switches to a serial mode independent of the Ipython cluster.
        src_in_tree determines whether the source or target population will go into the tree.
        If ĺeft blank, the bigger point cloud will be used, as this is faster.'''

        if not deparallelize:
            # set up client and workers
            import ipyparallel as ipp
            import os

            self.rc = ipp.Client()
            self.dv = self.rc[:]
            self.dv.use_cloudpickle() #this is necessary so that closures can be sent to the workers
            self.lv = self.rc.load_balanced_view()
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
                from . import parallel_util
        else:
            from . import parallel_util

        # Using cloudpickle, the variables used in the worker methods have to be in the workspace
        spts = self.spts
        tpts = self.tpts
        c_rad = self.c_rad
        src_in_tree = self.src_in_tree
        prefix = self.prefix

        print('sit', src_in_tree)

        lam_qpt = lambda ids: parallel_util.find_connections_3dpar(kdt, spts, tpts, c_rad, src_in_tree, ids, use_distance, avoid_self)

        if src_in_tree:
            nqpts = tpts.npts
        else:
            nqpts = spts.npts

        if not deparallelize:
            #chunk the query points so that each worker gets roughly the same amount of points to query in the KDTree
            #Only IDs are sent in order to speed it up by preallocating the coordinates and as much information as possible
            #to the workers.
            id_sp = int(np.ceil(nqpts/len(self.rc.ids)))
            id_ar = [np.arange(id_sp) + i * id_sp for i in range(int(np.ceil(nqpts/id_sp)))]
            id_ar[-1] = id_ar[-1][id_ar[-1]<nqpts]
            id_ar = [[i, id_ar[i]] for i in range(len(id_ar))]
            s = list(self.lv.map(lam_qpt, id_ar, block=True))
        else:
            s = lam_qpt([0, np.arange(nqpts)])

        print('Exited process, saving as:', prefix)


class Connect_2D(object):
    '''Finds connections between one structure that can be projected to a 2D plane and another structure
    that is represented by 3D coordinate points.'''

    def __init__(self, qpts_src, qpts_tar, c_rad):
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

        lam_qpt = lambda ids: parallel_util.find_connections_2dpar(kdt, pts, lpts, c_rad, lin_axis, lin_in_tree, lin_is_src, ids, debug)

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
                from . import parallel_util
            s = list(self.lv.map(lam_qpt, id_ar, block=True))
        else:
            # essentially the same as connections_pseudo_parallel
            from . import parallel_util
            s = []
            for id1 in tqdm(id_ar):
                s.append(lam_qpt(id1))

        self.result = pd.concat(s, ignore_index=True)

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

    def save_result(self, prefix='', table_name='connection', save_mode='sqlite'):
        '''Saves the results as produced by the query_x_in_y method similarly as BREP.
        res = result(containing a list of arrays/lists with the found IDs
            -> first index = query point ID
            -> numbers within the 2nd level arrays/lists -> tree point IDs
        res_l = for the linear point species, distance from the first outer boder
        prefix = prefix for the result file -> path, output specifier
        With the two last bools different 4 different modes can be created to have flexibility.'''

        prefix = str(prefix)
        if save_mode == 'sqlite':
            import sqlite3
            conn = sqlite3.connect(prefix + '.db')
            c = conn.cursor()
            warnings.warn('Pre-existing table ' + table_name + ' will be destroyed.')
            c.execute('DROP TABLE IF EXISTS ' + table_name)
            conn.commit()
            conn.close()

        print('Begin writing the results.')
        if save_mode == 'sqlite':
            foutname = prefix+'.db'
            conn = sqlite3.connect(foutname)
            self.result.to_sql(table_name, conn, if_exists='append', index=False)
            conn.close()

        if save_mode == 'hdf':
            foutname = prefix+'.h5'
            store = pd.HDFStore(foutname, 'w')
            store.append(table_name, self.result)
            store.close()

        print('Done writing, saving as: {}.'.format(prefix))
