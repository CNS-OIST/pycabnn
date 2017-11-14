import argparse 
import numpy as np
import datetime
import neuron
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from neuron import hoc, h




# On the naming conventions: I tried to keep variable names similar as the ones in the brep.scm file
# However, all _ had to bee transformed to _ , and sometimes a postfix like _fn (filename) or _num (number) was added for clarification.

class Connector (object):
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='BREP for py ', conflict_handler='resolve')
        self.parser.add_argument("--config_fn",   type = str, default = '', help ="use the given hoc configuration file to obtain parameters")
        self.parser.add_argument("--gc_points_fn",  type = str, default = '', help ="load originating points for granule cells from given file, default is randomly generated")
        self.parser.add_argument("--gct_points_fn", type = str, default = '', help ="load junction points for parallel fibers from given file, default is offset from GC soma points")
        self.parser.add_argument("--goc_points_fn", type = str, default = '' , help ="load originating points for Golgi cells from given file, default is randomly generated") #!!
        self.parser.add_argument("--rng_seeds",     type =list, default = [13, 17, 19, 23, 29, 37], help ="Use the given seeds for random number generation")
        self.parser.add_argument("--mpi_split",     type = int, default = 0, help ="perform MPI split operation") #actually it should be color, and only if this element exists, the communicator is split. Thus, not sure if I put the right default.
        
        self.parser.add_argument("--x_extent",  type = float, default = 1200.0, help ="X_extent of patch")
        self.parser.add_argument("--y_extent",  type = float, default = 300.0 , help ="Y_extent of patch")
        self.parser.add_argument("--z_extent",  type = float, default = 150.0 , help ="Z_extent of patch")
                                 
        self.parser.add_argument("--pf_length", type = float, default = 100.0, help ="parallel fiber length")
        self.parser.add_argument("--pf_step",   type = float, default = 30.0 , help ="parallel fiber step size")
        self.parser.add_argument("--pf_start",  type = int,   default = 0,     help ="starting index for parallel fiber points")                                     
        self.parser.add_argument("--aa_length", type = float, default = 200.0, help ="ascending axon length")
        self.parser.add_argument("--aa_step",   type = float, default = 100.0, help ="ascending axon step size")

        self.parser.add_argument("--goc_num", type = int , default = 200, help ="number of Golgi cells")
        self.parser.add_argument("--goc_mean_distance", type = float , default = 50.0, help ="mean distance between Golgi cells and Golgi cell grid")
        self.parser.add_argument("--gc_num",    type = int,   default = 10000, help ="number of granule cells and parallel fibers")
                                 
        self.parser.add_argument("--goc_start",        type = int,   default = 0,    help ="starting index for Golgi cell points default is 0")
        self.parser.add_argument("--pf_goc_zone",      type = float, default = 5.0,  help ="PF to Golgi cell connectivity zone" ) # verify datatype, not sure what the zone parameters mean so far.
        self.parser.add_argument("--aa_goc_zone",      type = float, default = 5.0,  help ="AA to Golgi cell connectivity zone" )  
        self.parser.add_argument("--goc_goc_zone",     type = float, default = 30.0, help ="Golgi to Golgi cell connectivity zone" )
        self.parser.add_argument("--goc_goc_gap_zone", type = float, default = 30.0, help ="Golgi to Golgi gap junction connectivity zone")
                                 
        #GOC dendrites
        self.parser.add_argument("--goc_dendrites",               type = int,   default = 4,     help ="number of Golgi cell dendrites")                                 
        
        self.parser.add_argument("--goc_theta_apical_min",        type = float, default = 30.0,  help ="min angle used to determine height of apical dendrite in z direction")
        self.parser.add_argument("--goc_theta_apical_max",        type = float, default = 60.0,  help ="max angle used to determine height of apical dendrite in z direction")
        self.parser.add_argument("--goc_theta_apical_stdev",      type = float, default = 1.0,   help ="stdev of angle used to determine height of apical dendrite in z direction")
        
        self.parser.add_argument("--goc_theta_basolateral_min",   type = float, default = 30.0,  help ="min angle used to determine height of basolateral dendrite in z direction")
        self.parser.add_argument("--goc_theta_basolateral_max",   type = float, default = 60.0,  help ="max angle used to determine height of basolateral dendrite in z direction")
        self.parser.add_argument("--goc_theta_basolateral_stdev", type = float, default = 1.0,   help ="stdev of angle used to determine height of basolateral dendrite in z direction")
        
        self.parser.add_argument("--goc_apical_dendheight",       type = float, default = 100.0, help ="height of Golgi cell apical dendritic cone")
        self.parser.add_argument("--goc_apical_radius",           type = float, default = 100.0, help ="radius of Golgi cell apical dendrite cone")
        self.parser.add_argument("--goc_apical_nseg",             type = int,   default = 2,     help ="number of segments of GoC apical dendrite")
        self.parser.add_argument("--goc_apical_nsegpts",          type = int,   default = 4,     help ="number of points per segments of GoC apical dendrite")
        
        self.parser.add_argument("--goc_basolateral_dendheight",  type = float, default = 100.0, help ="height of Golgi cell basolateral dendritic cone")
        self.parser.add_argument("--goc_basolateral_radius",      type = float, default = 100.0, help ="radius of Golgi cell basolateral dendrite cone")
        self.parser.add_argument("--goc_basolateral_nseg",        type = int,   default = 2,     help ="number of segments of GoC basolateral dendrite")
        self.parser.add_argument("--goc_basolateral_nsegpts",     type = int,   default = 4,     help ="number of points per segments of GoC basolateral dendrite")
        
        #GOC axons
        self.parser.add_argument("--goc_axons",      type = int,   default = 10,    help ="number of Golgi cell axons")
        self.parser.add_argument("--goc_axonsegs",   type = int,   default = 1,     help ="number of Golgi cell axon segments")
        self.parser.add_argument("--goc_axonpts",    type = int,   default = 2,     help ="number of Golgi cell axon points")
        self.parser.add_argument("--goc_axon_x_min", type = float, default =-200.0, help ="minimum extent of Golgi cell axons along X axis")
        self.parser.add_argument("--goc_axon_x_max", type = float, default = 200.0, help ="maximum extent of Golgi cell axons along X axis")
        self.parser.add_argument("--goc_axon_y_min", type = float, default =-200.0, help ="minimum extent of Golgi cell axons along Y axis")
        self.parser.add_argument("--goc_axon_y_max", type = float, default = 200.0, help ="maximum extent of Golgi cell axons along Y axis")
        self.parser.add_argument("--goc_axon_z_min", type = float, default =-30.0,  help ="minimum extent of Golgi cell axons along Z axis")
        self.parser.add_argument("--goc_axon_z_max", type = float, default =-200.0, help ="maximum extent of Golgi cell axons along Z axis")
        
        #Output utilities
        self.parser.add_argument("--save_aa", type = bool, default = False, help ="save ascending axon points")
        self.parser.add_argument("--save_pf", type = bool, default = False, help ="save parallel fiber points")
        self.parser.add_argument("--output",  type = str,  default = '',    help ="specify output file prefix, default is a timestemp.") # we should set a default here that puts in a timeprint in order to prevent unvoluntary overwrite
        self.parser.add_argument("--verbose", type = bool, default = False, help ="verbose model")
        self.config_dict = {
        'y_extent' : 'GoCyrange', 
        'goc_apical_nsegpts' : 'GoC_Ad_nsegpts', 
        'goc_theta_basolateral_min' : 'GoC_Btheta_min', 
        'goc_basolateral_nseg' : 'GoC_Bd_nseg', 
        'goc_basolateral_nsegpts' : 'GoC_Bd_nsegpts', 
        'goc_axon_z_min' : 'GoC_Axon_Zmin', 
        'goc_axon_y_max' : 'GoC_Axon_Ymax', 
        'num_gc' : 'numGC', 
        'goc_basolateral_dendheight' : 'GoC_PhysBasolateralDendH', 
        'aa_goc_zone' : 'AAtoGoCzone', 
        'mean_goc_distance' : 'meanGoCdistance', 
        'goc_theta_apical_min' : 'GoC_Atheta_min', 
        'goc_dendrites' : 'numDendGolgi', 
        'goc_axon_x_min' : 'GoC_Axon_Xmin', 
        'goc_theta_basolateral_stdev' : 'GoC_Btheta_stdev', 
        'goc_theta_apical_stdev' : 'GoC_Atheta_stdev', 
        'goc_theta_apical_max' : 'GoC_Atheta_max', 
        'goc_axonsegs' : 'GoC_Axon_nseg', 
        'num_goc' : 'numGoC', 
        'goc_axonpts' : 'GoC_Axon_npts', 
        'z_extent' : 'GoCzrange', 
        'goc_axons' : 'numAxonGolgi', 
        'goc_theta_basolateral_max' : 'GoC_Btheta_max', 
        'pf_step' : 'PFstep', 
        'aa_step' : 'AAstep', 
        'x_extent' : 'GoCxrange', 
        'goc_apical_radius' : 'GoC_PhysApicalDendR', 
        'goc_axon_x_max' : 'GoC_Axon_Xmax', 
        'goc_basolateral_radius' : 'GoC_PhysBasolateralDendR', 
        'goc_goc_gap_zone' : 'GoCtoGoCgapzone', 
        'goc_apical_nseg' : 'GoC_Ad_nseg', 
        'goc_axon_y_min' : 'GoC_Axon_Ymin', 
        'goc_axon_z_max' : 'GoC_Axon_Zmax', 
        'goc_apical_dendheight' : 'GoC_PhysApicalDendH', 
        'goc_goc_zone' : 'GoCtoGoCzone', 
        'pf_goc_zone' : 'PFtoGoCzone', 
        'pf_length' : 'PFlength' }

        
    def init_from_script (self, cml_str = ['']):
        self.args = self.parser.parse_args(cml_str)
        d  =  {}
        for k_c in range(len(cml_str)-1):
            if len(str(cml_str[k_c])) > 2:
                if cml_str[k_c][:2] == '--':
                    d[cml_str[k_c][2:]] = cml_str[k_c+1]
        setattr (self, 'cl_args', d)
                
            
  
    def init_from_cl(self):
        self.args = self.parser.parse_args()
        from collections import defaultdict
        # to do: Check!!
        d=defaultdict(list)
        for k, v in ((k.lstrip('-'), v) for k,v in (a.split('=') for a in sys.argv[1:])):
            d[k].append(v)
        setattr (self, 'cl_args', d)
        #https://stackoverflow.com/questions/12807539/how_do_you_convert_command_line_args_in_python_to_a_dictionary
    
    @classmethod    
    def check_output_prefix(self):
        '''checks if there is a specified output prefix.
        If not, will generate one from the timestampe'''
        if self.args.output == '':
            #! mkdir 'Res_{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
            self.args.output = 'Res_{:%Y_%m_%d_%H_%M_%S}/'.format(datetime.datetime.now())
            
    def read_in_config(self, overwrite_config = True):
        '''checks if a config file has been specified, and if so, updates the args.
        If overwrite_config is set True, command line arguments that specify a parameter that also exists
        in the config file will have priority.'''
        if self.args.config_fn == '':
            warnings.warn('Cannot find config file!')
        else:  #read in config file using neurons h object
            self.p_verb('Reading config file')
            neuron.h.xopen(self.args.config_fn)
            d_l = dir(neuron.h) #get set attributes from the config file
            c_d = dict((v,k) for k,v in self.config_dict.items()) #exchange key and value

            for h_k in d_l: #check for relevant attributes that are set in the config file and update them
                if h_k in c_d.keys() and h_k not in self.cl_args.keys():
                    if hasattr (self.args, c_d[h_k]):
                        self.p_verb('Configurated {}: Was {}, now is {}'.format(c_d[h_k], getattr (self.args, c_d[h_k]), getattr (neuron.h, h_k)))
                        setattr (self.args, c_d[h_k], getattr (neuron.h, h_k))
                    else:
                        print ('Did not find {}'.format(c_d[h_k]))
                #deal with parameters that are defined double
                elif h_k in c_d.keys() and h_k in self.cl_args.keys():
                    if hasattr (self.args, c_d[h_k]):
                        if overwrite_config:
                            warnings.warn('Parameter {} was set both by command line and in config, will use value from command line'.format(c_d[h_k]))
                        else:
                            warnings.warn('Parameter {} was set both by command line and in config, will use value from config file'.format(c_d[h_k]))
                            setattr (self.args, c_d[h_k], getattr (neuron.h, h_k))
                
            # The following two parameters are an exception:
            if 'GLdepth' in d_l and 'PCLdepth' in d_l and not 'aa-length' in self.cl_args.keys():
                setattr (self.args, 'aa-length', getattr(neuron.h, 'GLdepth')+getattr(neuron.h,'PCLdepth'))
    






            
    
    def p_verb (self, stat, *args):
        '''prints statement only if the print mode is on.
        Prints args in some smart ways'''
        if self.args.verbose:
            print (stat)

    
    def get_GC_Points():
        ''' 
        corresponds to GC_Points statement in brep.csm file.
        Read in the GC_Points provided in a file or render some randomly. (UniformRandomProcess)
        '''
        pass
        
    def get_GCT_Points():
        '''
        corresponds to the GCT_Points statement from the 
        '''
    
    def get_GOC_Points():
        '''
        get GOC Points from file or render them
        '''
        

class Cell_pop (object):

    def __init__(self, my_args):
        self.args = my_args


    def load_somata(self, fn_or_ar):
        if type(fn_or_ar) == str:
            try: self.read_in_soma_file(fn_or_ar)
            except: print ('Tried to read in ', fn_or_ar, ', failed.')
        else:
            try:
                if fn_or_ar.shape[len(fn_or_ar.shape)-1] == 3:
                    self.som = fn_or_ar
                else: print ('Cannot recognize array as coordinates')
            except: print ('Could not read in soma points')

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





    # Should have:
    # Points
    # Cell numbers
    # Segments
class Query_point (object):
    def __init__ (self, coord, IDs = [], segs = []):
        self.coo = coord
        self.idx = IDs
        self.seg = segs
        self.npts = len(coord)

    def lin_check (self):
        pass
#class Query_point_lin (Query_point):






class Connect_2D(object):
    def __init__(self, qpts1, qpts2, c_rad):
        self.pts1 = qpts1
        self.pts2 = qpts2
        self.c_rad = c_rad




    def construct_tree(self):
        pass




class Golgi_pop (Cell_pop):
    def __init__(self, my_args):
        Cell_pop.__init__(self,my_args)



    def add_dendrites(self):
        a_rad = self.args.GoC_PhysApicalDendR
        a_h = self.args.GoC_PhysApicalDendH
        a_ang = [self.args.GoC_Atheta_min, self.args.GoC_Atheta_max]
        a_std = self.args.GoC_Atheta_stdev
        a_n = int(self.args.GoC_Ad_nseg * self.args.GoC_Ad_nsegpts) # numbr of points per dencrite

        b_rad = self.args.GoC_PhysBasolateralDendR
        b_h = self.args.GoC_PhysBasolateralDendH
        b_ang = [self.args.GoC_Btheta_min, self.args.GoC_Btheta_max]
        b_std = self.args.GoC_Btheta_stdev
        b_n = int(self.args.GoC_Bd_nseg * self.args.GoC_Bd_nsegpts)

        a_dend, a_idx, a_sgts = self.gen_dendrite(a_rad, a_h, a_ang, a_std, a_n)
        b_dend, b_idx, b_sgts = self.gen_dendrite(b_rad, b_h, b_ang, b_std, b_n)

        def conc_ab (a, b):
            def flatten_cells (dat):
                if len(dat.shape) == 2: dat = np.expand_dims(dat, axis = 2)
                return dat.reshape(dat.shape[0]*dat.shape[1],dat.shape[2])
            return np.concatenate((flatten_cells(a), flatten_cells(b)))

        all_dends = conc_ab (a_dend, b_dend)
        all_idx = conc_ab (a_idx, b_idx)
        all_sgts = conc_ab (a_idx, b_idx)

        self.a_dend = a_dend
        self.b_dend = b_dend
        self.qpts = Query_point (all_dends, all_idx, all_sgts)


    def save_dend_coords(self):
        pass

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
        sgts: shape is #cells x #pts -> segment for each point, starts at 0
        where #pts = #segment per dendrite x# dendrites generated with this function
        '''
        c_gr = np.linspace(0,1,c_n)*np.ones((3, c_n)) #linspace grid between 0 and 1 with c_n elements
        b_res = []
        idx = []  #cell indices
        segs = [] #segment number
        for i in range(len(self.som)): #each cell
            som_c = self.som[i,:]
            d_res = []
            d_segs = []
            for cc_m in c_m: #each dendrite
                ep_ang = (np.random.randn()*c_std + cc_m)*np.pi/180 #angle
                pt = ([np.sin(ep_ang)*c_r, np.cos(ep_ang)*c_r, c_h])*c_gr.T #coordinates of the dendrite = endpoint*grid 
                d_res = d_res + list(pt+som_c)
                d_segs = d_segs + list(np.arange(c_n))
            b_res.append(np.array(d_res))
            segs.append(d_segs)
            idx.append((np.ones(len(d_res))*i).astype('int'))
        return np.array(b_res), np.array(idx), np.array(segs)





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
        pf_length = self.args.PFlength
        try:
            self.pf_dots = self.aa_dots.copy()
            self.pf_dots[:,0,2] = self.pf_dots[:,1,2] #z axis shall be the same
            self.pf_dots[:,0,0] = self.pf_dots[:,0,0] - pf_length/2
            self.pf_dots[:,1,0] = self.pf_dots[:,1,0] + pf_length/2
            self.qpts_pf = Query_point(self.pf_dots)
        except Exception as e:
            #raise e
            print ('I need ascending axon points to calculate the parallel fibers')

    def add_3D_aa_and_pf(self):
        aa_length = self.args.PCLdepth+ self.args.GLdepth
        aa_nd = int(self.aa_length / self.args.AAstep) #number of dots for the aa
        aa_sp = np.linspace(0, self.aa_length, aa_nd)

        pf_nd = int(self.args.PFlength/self.args.PFstep) # number of dots for the pf
        pf_sp = np.linspace(-self.args.PFlength/2, self.args.PFlength/2)

        self.aa_dots = np.zeros((len(coo), aa_nd, 3))
        self.pf_dots = np.zeros((len(coo), pf_nd, 3))
        aa_idx = np.zeros((len(coo), aa_nd))
        aa_sgts= np.zeros((len(coo), aa_nd))
        pf_idx = np.zeros((len(coo), pf_nd))
        pf_sgts= np.zeros((len(coo), pf_nd))
        for i, som in enumerate(coo):
            self.aa_dots[i] = np.ones((aa_nd, 3))*som
            self.aa_dots[i,:,2] = aa_dots[i,:,2] + aa_sp
            aa_idx[i,:] = i
            aa_sgts[i,:] = np.arange(aa_nd)
            self.pf_dots[i] = np.ones((pf_nd,3))*aa_dots[i,-1, :]
            self.pf_dots[i,:,0] = pf_dots[i,:,0] + pf_sp
            pf_idx[i,:]  = i
            pf_sgts[i,:] = np.arange(pf_nd) #! Not necessarily nice

        self.qpts_aa = Query_point(self.aa_dots, aa_idx, aa_sgts)
        self.qpts_pf = Query_point(self.pf_dots, pf_idx, pf_sgts)



