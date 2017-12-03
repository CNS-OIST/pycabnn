import argparse 
import numpy as np
import datetime
#import neuron
import csv
import warnings
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree 

#try: from neuron import hoc, h
#except: pass
    #print ('No neuron installation found! Will try to run using Pseudo_hoc object.')




class Pseudo_hoc (object):
    '''Up to now, the program depends on a hoc object that contains the parameters for the simulation.
    However, as in the cluster, neuron is not installed for python 3, this class is a workaround:
    First, a dict has to be generated (and probably pickled) from the hoc file in a python distribution that has neuron installed.
    This dict (or the file containing it) is then read in in this class, and an empty object with no other functionalities gets
    assigned all the parameters from the dict as attributes. The resulting object can then be used as a parameter carrier just as the hoc file.'''
    
    def __init__ (self, ad_or_fn = []):
        '''Add parameters from dict or filename to pseudo_hoc object'''
        #Make a pseudo-hoc object
        if ad_or_fn == []: return
        elif type(ad_or_fn) == str:
            try:
                import pickle
                with open (ad_or_fn, 'rb') as f_in:
                    ad_or_fn = pickle.load(f_in)
            except: print ('Tried to read in ', ad_or_fn, ' as a file, failed')
        else: assert type(ad_or_fn) == dict, 'Could not read in ' + ad_or_fn
        # Add all elements from the read in file as arguments to the pseudo-hoc object
        for k,v in ad_or_fn.items():
            #As for the pickling process, all values had to be declared as strings, try to convert them back to a number
            try: v = float(v) 
            except: pass
            try: setattr(self, k, v)
            except: pass
            
    def convert_hoc_to_pickle (self, config_fn, output_fn = 'pseudo_hoc.pkl'):
        '''Take a .hoc config file and pickle it as a neuron-independent python dict.'''
        try: import neuron
        except: 
            print ('Could not import neuron, go to a python environment with an installed neuron version and try again.')
            return
        neuron.h.xopen(config_fn)
        d = dir(h)
        h_dict = dict()
        #Transfer parameters from the hoc object to a python dictionary
        for n,el in enumerate(d):
            #note! so far all used parameters in the BREPpy started with capital letters, so only adding atrributes that start with capital letters is a reasonable filtering method.
            #However, this is to be kept in mind when adding additional parameters to the parameter file.
            if el[0].isupper():
                try:
                    #The value has to be converted to its string representation to get rid of the hoc properties.
                    #Must be kept in mind when reading in though.
                    h_dict[el] = repr(getattr(h,el))
                except: pass
        #Dump the dictionary
        import pickle
        with open(output_fn, 'wb') as f:
            pickle.dump(h_dict, f)



def str_l (ar): 
    '''make a space-seperated string from all elements in a 1D-array'''
    return (' '.join(str(ar[i]) for i in range(len(ar))))


# This is a test for Parallelization experimnets
def pt_in_tr2 (kdt, pt, c_rad):
    warnings.simplefilter('ignore')
    res, = kdt.query_radius(np.expand_dims(pt, axis = 0), r = c_rad)
    return res


def pts_in_tr_ids (kdt, q_pts, lax_c, lax_range, c_rad, ids, lin_in_tree):
           
    res = []
    l_res = []
    for i, pt in enumerate(q_pts[ids]): #iterate through the query points
        # find the points within the critical radius
        warnings.simplefilter('ignore')
        ind, = kdt.query_radius(np.expand_dims(pt, axis = 0), r = c_rad)
        try: 
            print (numpy.sqrt (12))
        except: pass
        try:
            print (np.sqrt (113))
        except: pass


        #check if the found points match along the linearized axis and if so, add distance from the beginning of the linearized axis
        if lin_in_tree: 
            ind = ind[numpy.logical_and(lax_range[ind,0]<=lax_c[i], lax_range[ind,1]>= lax_c[i])]
            l_res.append(lax_c[i] - lax_range[ind,0])
        else:
            ind = ind[numpy.logical_and(lax_range[i,0]<=lax_c[ind], lax_range[i,1]>= lax_c[ind]).ravel()]
            l_res.append(lax_c[ind] - lax_range[i,0])

        res.append(ind.astype('int'))

    return [res, l_res]






####################################################################
## CONNECTOR PART                                                 ##
####################################################################

#class Connect_3D_parallel (object):




class Connect_2D(object):

    def __init__(self, qpts_src, qpts_tar, c_rad, prefix = ''):

        #If the arguments are np arrays, make a Query_point from it.
        if type(qpts_src).__module__ == np.__name__ : qpts_src = Query_point(qpts_src)
        if type(qpts_tar).__module__ == np.__name__ : qpts_tar = Query_point(qpts_tar)

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
            print ('Failure to initialize connector, there is not one linearized and one regular point set')
        self.lin_is_src = qpts_src.lin
        self.c_rad = c_rad
        self.prefix = prefix


    def connections_serial (self):


        '''try:
            import ipyparallel as ipp
            rc = ipp.Client()
            dv = rc[:]
            lv = rc.load_balanced_view()
            print ('Started parallel process with', len(rc.ids), 'workers.')
        except:
            print ('Parallel process could not be started!')
            if True:
                print ('Will do it sequentially instead...')
                res, l_res = self.find_connections()
                self.save_results (res, res_l, self.prefix)
            return'''


        import ipyparallel as ipp
        self.rc = ipp.Client()
        self.dv = self.rc[:]
        self.dv.use_cloudpickle()
        self.lv = self.rc.load_balanced_view()

        import os
        print(self.rc[:].apply_sync(os.getcwd))
        print (os.getcwd())






        print ('Started parallel process with', len(self.rc.ids), 'workers.')

        kdt, q_pts = self.get_tree_setup (return_lax = False)
        import cloudpickle
        self.dv.block = True
        with self.dv.sync_imports():
            import parallel_util

        con2d_dict = dict (
            kdt = kdt,
            q_pts = q_pts,
            c_rad = self.c_rad,
            lin_axis = self.lin_axis,
            lin_in_tree = self.lin_in_tree,
            lin_is_src = self.lin_is_src,
            prefix = self.prefix,
            pts = self.pts.coo,
            lpts = self.lpts.coo)
        self.dv.push (con2d_dict)

        pts = self.pts
        lpts = self.lpts
        c_rad = self.c_rad
        lin_axis = self.lin_axis
        lin_in_tree = self.lin_in_tree
        lin_is_src = self.lin_is_src
        prefix = self.prefix




        lam_qpt = lambda ids: parallel_util.find_connections_2dpar (kdt, pts, lpts, c_rad, lin_axis, lin_in_tree, lin_is_src, ids, prefix)
        self.dv.block = False
        '''def get_id_array (len_id, id_sp, add_num = True):
            id_sp = int(id_sp)
            c = [np.arange(id_sp) + i * id_sp for i in range(int(np.ceil (len_id/id_sp)))]
            c[-1] = c[-1][c[-1]<len_id]
            if add_num: c= [[i, c[i]] for i in range(len(c))]
            return c'''

        id_sp = int(np.ceil(len(q_pts)/len(self.rc.ids)))
        id_ar = [np.arange(id_sp) + i * id_sp for i in range(int(np.ceil (len(q_pts)/id_sp)))]
        id_ar[-1] = id_ar[-1][id_ar[-1]<len(q_pts)]
        id_ar= [[i, id_ar[i]] for i in range(len(id_ar))]



        

        s = list(self.lv.map (lam_qpt, id_ar, block = True))
         # Note that this can also be set False, but in the current version this gives an error, compare: https://github.com/ipython/ipyparallel/issues/274
        print ('whoa')
        s = list(self.lv.apply (lam_qpt, id_ar, block = True))




    def get_tree_setup (self, return_lax = True, lin_in_tree = []):
        '''Gets the setup for the connection.
        lin_in_tree determines whether
         '''

        # if not specified which of the point population is the query point population, take the smaller one and put the bigger one in the tree.
        if not type(lin_in_tree).__name__ == 'bool':
            lin_in_tree = self.lpts.npts > self.pts.npts
        self.lin_in_tree = lin_in_tree
        #depending on which points shall be in the tree and which shall be the query points, assign, and project at the same time
        if lin_in_tree:
            tr_pts = self.lpts.coo[:,0,np.invert(self.lin_axis)]
            q_pts = self.pts.coo[:, np.invert(self.lin_axis)]  
        else:
            q_pts = self.lpts.coo[:,0,np.invert(self.lin_axis)]
            tr_pts = self.pts.coo[:, np.invert(self.lin_axis)] 
                              

        #build 2D Tree
        kdt = KDTree(tr_pts)

        if not return_lax: return kdt, q_pts
        else: 
            #get the information for the axis along which the projection is done
            lax_c = self.pts.coo[:,self.lin_axis] 
            lax_range = self.lpts.coo[:,:,self.lin_axis] 
            lax_range = lax_range.reshape((lax_range.shape[0], lax_range.shape[1]))
            return kdt, q_pts, lax_c, lax_range, self.lin_in_tree


    def find_connections (self, lin_in_tree = []):
        
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
                l_res.append(lax_c[i] - lax_range[ind,0] + self.lpts.lin_offset[ind])
            else:
                ind = ind[np.logical_and(lax_range[i,0]<=lax_c[ind], lax_range[i,1]>= lax_c[ind]).ravel()]
                l_res.append(lax_c[ind] - lax_range[i,0] + self.lpts.lin_offset[i])

            res.append(ind.astype('int'))

        return res, l_res


    def save_results (self, res, res_l, prefix = ''):
        '''Saves the results as produced by the query_x_in_y method similarly as BREP.
        res = result (containing a list of arrays/lists with the found IDs 
            -> first index = query point ID
            -> numbers within the 2nd level arrays/lists -> tree point IDs
        res_l = for the linear point species, distance from the first outer boder
        prefix = prefix for the result file -> path, output specifier
        With the two last bools different 4 different modes can be created to have flexibility.'''

        # file names
        fn_tar = prefix + 'target.dat'
        fn_src = prefix + 'source.dat'
        fn_segs = prefix + 'segments.dat'
        fn_dis = prefix + 'distance.dat'

        with open (fn_tar, 'w') as f_tar, open (fn_src, 'w') as f_src, open (fn_dis, 'w') as f_dis, open (fn_segs, 'w') as f_segs: 

            for l, (cl, cl_l) in enumerate(zip(res, res_l)):
                
                assert len(cl) == len(cl_l), 'Something went wrong, all corresponding lists in your input arguments should have the same length'               
                assert hasattr(self, 'lin_in_tree'), 'Attribute lin_in_tree must be set, this should happen in the get_tree_setup method!'
                assert hasattr(self, 'lin_is_src'), 'Attribute lin_is_src must be set, this should happen at initialization depending on the order of the query point arguments!'

                if len(cl_l)>0:
                    f_dis.write("\n".join(map(str, cl_l)))
                    #first, get the cell IDS of the query and tree points (for the linear points, that is just the point ID, 
                    #for the other points this information has to be extracted from the corresponding Query_points object.
                    #Segments also corresponds to the 3D point population, right value is acquired from Query-points object.
                    if self.lin_in_tree: 
                        s_ar = self.pts.seg[l,:].astype('int')
                        f_segs.write("\n".join(map(str_l, [s_ar for i in range (len(cl))])))#*np.ones((len(cl), len (s_ar)))))) 

                        q_id = np.ones(len(cl))*self.pts.idx[l]
                        tr_id = cl
                    else:
                        f_segs.write("\n".join(map(str_l, self.pts.seg[cl].astype('int'))))
                        q_id = self.pts.idx[cl] 
                        tr_id = np.ones(len(cl))*l 

                    #depending on which population should be source and which should be target, save cell IDs accordingly.
                    if self.lin_in_tree == self.lin_is_src:
                        f_tar.write("\n".join(map(str, tr_id)))
                        f_src.write("\n".join(map(str, q_id)))
                    else:
                        f_tar.write("\n".join(map(str, q_id)))
                        f_src.write("\n".join(map(str, tr_id )))

                    #need to attach one more line here or we get two elements per line 
                    f_dis.write("\n")
                    f_src.write("\n")
                    f_tar.write("\n")
                    f_segs.write("\n")


#class Connect_3D(object):


class Query_point (object):
    def __init__ (self, coord, IDs = [], segs = [], lin_offset = 0):
        self.coo = coord
        self.seg = segs
        self.npts = len(coord)
        if not IDs == []: self.idx = IDs
        else: self.idx = np.arange(len(coord)) 
        self.lin = self.lin_check()
        if self.lin: #lin_offset is added to the distance for each cell.
            try: lin_offset = float (np.array(lin_offset)) * np.ones(self.npts)
            except:
                assert (len(lin_offset) == self.npts), 'lin_offset should be a scalar or an array with length npts!'
            finally: self.lin_offset = lin_offset


    def lin_check (self):
        if len(self.coo.shape) == 3:
            if self.coo.shape[1] == 2 and self.coo.shape[2] == 3:
                sm = sum(abs(self.coo[:,0,:] - self.coo[:,1,:]))
                no_dif = [np.isclose(sm[i],0) for i in range(len(sm))]
                if sum(no_dif) == 2: # 2 coordinates are the same, one is not
                    self.lin_axis = np.invert(no_dif) #this one is the axis that cn be linearized
                    return True
        return False


    def linearize (self):
        pass
        #this function should linearize points when they are in a higher structure than nx3, and the IDs and 


####################################################################
## POPULATION PART                                                ##
####################################################################

class Cell_pop (object):

    def __init__(self, my_args):
        self.args = my_args

    def load_somata(self, fn_or_ar):
        ''' Loads somata and adds them to the population object:
        Either from a coordinate file (if fn_or_ar is a filename)
        Or diectly from a numpy array (if fn_or_ar is such)'''
        if type(fn_or_ar) == str:
            try: self.read_in_soma_file(fn_or_ar)
            except: print ('Tried to read in ', fn_or_ar, ', failed.')
        else:
            try:
                if fn_or_ar.shape[len(fn_or_ar.shape)-1] == 3:
                    self.som = fn_or_ar
                else: print ('Cannot recognize array as coordinates')
            except: print ('Could not read in soma points')


    def save_somata(self, prefix = '', fn = ''):
        if fn == '': fn = type(self).__name__ + '_coords.dat'
        '''Save the soma coordinates'''
        assert hasattr(self, 'som'), 'Cannot save soma coordinates, as apparently none have been added yet'
        with open (prefix+fn, 'w') as f_out:
            f_out.write("\n".join(map(str_l, self.som)))



    def read_in_soma_file (self, fn, parse_ignore = True):
        ''' Reads in files such as the ones that BREP returns.
        Represents lines as rows, nth element in each line as column.
        fn = Filename
        parse_ignore: If something cannot be parsed, it will be ignored. If this parameter is set false, it will complain
        returns: 2d-array of floats'''
        res = []
        with open (fn, 'r', newline = '') as f:
            rr = csv.reader(f, delimiter = ' ')
            err = [] # list of elements that could not be read in
            for line in rr: # lines -> rows
                ar = []
                for j in range(len(line)): 
                    try: ar.append(float(line[j]))
                    except: err.append(line[j])
                res.append(np.asarray(ar))
        if len(err)> 0 and not parse_ignore: print ('Could not parse on {} instances: {}'.format(len(err), set(err)))
        self.som = np.asarray(res)
        self.n_cell = len(self.som)


    def coord_reshape (dat, n_dim = 3):
        ''' Reshapes coordinate files with several points in one line by adding an extra axis.
        Thus, converts from an array with shape (#cells x (#pts*ndim)) to one with shape (#cells x #pts x ndim)'''
        dat = dat.reshape([dat.shape[0], int(dat.shape[1]/n_dim),n_dim])
        return dat


    def plot_somata (self, new_fig = True, *args, **kwargs):
        if new_fig:
            plt.figure()
        ax = plt.gcf().gca(projection='3d')
        ax.plot(self.som[:,0], self.som[:,1], self.som[:,2], *args, **kwargs)


    def gen_random_cell_loc (self, n_cell):
        '''Random generation for cell somatas
        n_cell (int) = number of cells
        Gc_x, Gc_y, Gc_z (float) = dimensions of volume in which cells shall be distributed
        Algorithm will first make a grid that definitely has more elements than n_cell
        Each grid field is populated by a cell, then those cells are displaced randomly
        Last step is to prune the volume, i.e. remove the most outlying cells until the goal number of cells is left
        Returns: cell coordinates'''
        # get spacing for grid:
        vol_c = Gc_x*Gc_y*Gc_z/n_cell
        sp_def = vol_c**(1/3)/2
        
        #Get grid with a few too many elements
        gr = np.asarray([[i,j,k] 
                         for i in np.arange(0, Gc_x, 2*sp_def)   
                         for j in np.arange(0, Gc_y, 2*sp_def) 
                         for k in np.arange(0, Gc_z, 2*sp_def)])
        #random displacement
        grc = gr + np.random.randn(*gr.shape)*sp_def
        
        #then remove the ones that lie most outside to get the correct number of cells:
        #First find the strongest outliers
        lower = grc.T.ravel()
        upper = -(grc-[Gc_x, Gc_y, Gc_z]).T.ravel()
        most_out_idx = np.mod(np.argsort(np.concatenate((lower,upper))), len(grc))
        #In order to find the right number, must iterate a bit as IDs may occur twice (edges)
        del_el = len(grc) - n_cell # number of elements to be deleted
        n_del = del_el
        while len(np.unique(most_out_idx[:n_del])) < del_el:
            n_del = n_del + del_el - len(np.unique(most_out_idx[:n_del]))
        #Deletion step
        grc = grc[np.setdiff1d(np.arange(len(grc)), most_out_idx[:n_del]),:]
        return grc



class Golgi_pop (Cell_pop):

    def __init__(self, my_args):
        Cell_pop.__init__(self,my_args)


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
        def conc_ab (a, b):
            def flatten_cells (dat): # keep the last dimension (3, for spatial coordinates), ravel the first two (ncell*npts)
                if len(dat.shape) == 2: dat = np.expand_dims(dat, axis = 2)
                return dat.reshape(dat.shape[0]*dat.shape[1],dat.shape[2])
            return np.concatenate((flatten_cells(a), flatten_cells(b)))

        #concatenated dendrite information (coords, cell indices, segment information)
        all_dends = conc_ab (a_dend, b_dend) 
        all_idx = conc_ab (a_idx, b_idx)
        all_sgts = conc_ab (a_sgts, b_sgts)

        self.a_dend = a_dend
        self.b_dend = b_dend
        self.qpts = Query_point (all_dends, all_idx, all_sgts)



    def save_dend_coords(self, prefix = ''):
        ''' Save the coordinates of the dendrites, BREP style 
        -> each line of the output file corresponds to one cell, and contains all its dendrites sequentially'''
        assert hasattr(self, 'a_dend') or hasattr(self, 'b_dend'), 'Could not find any added dendrites'
        if hasattr(self, 'a_dend'):
            with open(prefix +  'GoCadendcoordinates.dat', 'w') as f_out:
                f_out.write("\n".join(map(str_l, self.a_dend)))
        else: warnings.warn('Could not find apical dendrite')
        if hasattr(self, 'b_dend'):
            with open(prefix +  'GoCbdendcoordinates.dat', 'w') as f_out:
                f_out.write("\n".join(map(str_l, self.a_dend)))
        else: warnings.warn('Could not find basal dendrite')


    def gen_dendrite (self, c_r, c_h, c_m, c_std, c_n):
        '''Generates dendrites as described in the paper:
        c_r = maximal radius of cone
        c_h = height of cone
        c_m = mean angle for each dendrite (number of elements = number of dendrites per cell)
        c_std = standard deviation (degree) for the angle of the dendrite
        c_n = number of points
        Returns three arrays:
        res: shape is #cells x #pts x 3 (coords) -> coordinates of the points
        idx: shape is #cells x #pts -> cell ids of the points (starting at 0)
        sgts: shape is #cells x #pts x 2 -> each point consists of [# segment, # dendrite], both starting from 1
        -> to be conform with BREP, this has to be slightly modified, see add_dendrites function.
        where #pts = #segment per dendrite x# dendrites generated with this function
        '''
        c_gr = np.linspace(0,1,c_n)*np.ones((3, c_n)) #linspace grid between 0 and 1 with c_n elements
        b_res = []
        idx = []  #cell indices
        for i in range(len(self.som)): #each cell
            som_c = self.som[i,:]
            d_res = []
            if i == 0: d_sg = [] #segments, only have to be calculated once as they are the same for every cell
            for n,cc_m in enumerate(c_m): #each dendrite
                ep_ang = (np.random.randn()*c_std + cc_m)*np.pi/180 #individual angle
                pt = ([np.sin(ep_ang)*c_r, np.cos(ep_ang)*c_r, c_h])*c_gr.T #coordinates of the dendrite = endpoint*grid 
                d_res = d_res + list(pt+som_c)
                if i == 0: d_sg = d_sg + list(np.array([np.arange(c_n), np.ones(c_n)*(n+1)]).T)
            b_res.append(np.array(d_res))
            idx.append((np.ones(len(d_res))*i).astype('int'))
        segs = np.array([d_sg for k in range(i+1)]) #replicate segment information for each cell

        return np.array(b_res), np.array(idx), segs



class Granule_pop (Cell_pop):
    def __init__(self, my_args):
        Cell_pop.__init__(self, my_args)
        self.aa_length = self.args.PCLdepth+ self.args.GLdepth


    def add_aa_endpoints_random (self):
        '''Generate aa endpoints with a random aa length (end point will be somewhere in mol_range)'''
        mol_range = [self.aa_length, self.aa_length+self.args.MLdepth]
        self.aa_dots = np.array([np.array([self.som[i], self.som[i]]) for i in range(len(self.som))]) 
        self.aa_dots[:,1,2] = np.random.uniform(mol_range[0], mol_range[1], len(self.aa_dots[:,1,2]))
        self.qpts_aa = Query_point(self.aa_dots)


    def add_aa_endpoints_fixed(self):
        '''Generate aa endpoints with a random aa length (end point will be somewhere in mol_range)'''
        #aa_length = self.args.PFzoffset   #NOTE: This value exists, but in the BREP original file it is replaced by the other definition
        self.aa_dots = np.array([np.array([self.som[i], self.som[i]]) for i in range(len(self.som))]) 
        self.aa_dots[:,1,2] = self.aa_dots[:,1,2] + self.aa_length
        self.qpts_aa = Query_point(self.aa_dots)


    def add_pf_endpoints (self):
        ''' Addd the endpoints of the parallel fibers [begin, end] for each cell'''
        pf_length = self.args.PFlength
        assert hasattr(self, 'aa_dots'), 'Cannot add Parallel Fiber, add ascending axon first!'
        self.pf_dots = self.aa_dots.copy()
        self.pf_dots[:,0,2] = self.pf_dots[:,1,2] #z axis shall be the same
        self.pf_dots[:,0,0] = self.pf_dots[:,0,0] - pf_length/2
        self.pf_dots[:,1,0] = self.pf_dots[:,1,0] + pf_length/2
        self.qpts_pf = Query_point(self.pf_dots, lin_offset = self.aa_dots[:,1,2] - self.aa_dots[:,0,2])

    def add_3D_aa_and_pf(self):
        '''adds 3-dimensional coordinates for ascending axons and parallel fiber to the granule cell objects.
        Both AA and PF are represented by regularly spaced dots'''

        aa_length = self.args.PCLdepth+ self.args.GLdepth
        aa_nd = int(self.aa_length / self.args.AAstep) #number of dots for the aa
        aa_sp = np.linspace(0, self.aa_length, aa_nd) #grid that contains the spacing for the aa points

        pf_nd = int(self.args.PFlength/self.args.PFstep) # number of dots for the pf
        pf_sp = np.linspace(-self.args.PFlength/2, self.args.PFlength/2) # grid that contains spacing of po points

        self.aa_dots = np.zeros((len(coo), aa_nd, 3))
        self.pf_dots = np.zeros((len(coo), pf_nd, 3))
        aa_idx = np.zeros((len(coo), aa_nd))
        aa_sgts= np.zeros((len(coo), aa_nd))
        pf_idx = np.zeros((len(coo), pf_nd))
        pf_sgts= np.zeros((len(coo), pf_nd))

        for i, som in enumerate(coo):

            self.aa_dots[i] = np.ones((aa_nd, 3))*som #copy soma location for each point of the aa
            self.aa_dots[i,:,2] = aa_dots[i,:,2] + aa_sp # add the z offsets
            aa_idx[i,:] = i #cell indices, for the query object
            aa_sgts[i,:] = np.arange(aa_nd) #segment points, for the query object

            self.pf_dots[i] = np.ones((pf_nd,3))*aa_dots[i,-1, :] #uppermost aa point is the origin of the pf points
            self.pf_dots[i,:,0] = pf_dots[i,:,0] + pf_sp #this time, points differ only by their offset along the x direction
            pf_idx[i,:]  = i
            pf_sgts[i,:] = np.arange(pf_nd) #! Not necessarily nice

        self.qpts_aa = Query_point(self.aa_dots, aa_idx, aa_sgts)
        self.qpts_pf = Query_point(self.pf_dots, pf_idx, pf_sgts)


    def save_gct_points (self, prefix = ''):
        ''' Saves the coordinates of the Granule cell T points, i.e. the points where the Granule cell ascending axons
        split into the parallel fiber'''
        assert hasattr(self, 'aa_dots'),  'No ascending axons added yet'
        gctp = self.aa_dots[:,-1,:]
        with open (prefix+'GCTcoordinates.dat', 'w') as f_out:
            f_out.write("\n".join(map(str_l, gctp)))



