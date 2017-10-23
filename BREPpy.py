import argparse
import numpy as np
import datetime

# On the naming conventions: I tried to keep variable names similar as the ones in the brep.scm file
# However, all - had to bee transformed to _ , and sometimes a postfix like _fn (filename) or _num (number) was added for clarification.

class Brep (object):
    
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
        'y-extent' : 'GoCyrange', 
        'goc-apical-nsegpts' : 'GoC_Ad_nsegpts', 
        'goc-theta-basolateral-min' : 'GoC_Btheta_min', 
        'goc-basolateral-nseg' : 'GoC_Bd_nseg', 
        'goc-basolateral-nsegpts' : 'GoC_Bd_nsegpts', 
        'goc-axon-z-min' : 'GoC_Axon_Zmin', 
        'goc-axon-y-max' : 'GoC_Axon_Ymax', 
        'num-gc' : 'numGC', 
        'goc-basolateral-dendheight' : 'GoC_PhysBasolateralDendH', 
        'aa-goc-zone' : 'AAtoGoCzone', 
        'mean-goc-distance' : 'meanGoCdistance', 
        'goc-theta-apical-min' : 'GoC_Atheta_min', 
        'goc-dendrites' : 'numDendGolgi', 
        'goc-axon-x-min' : 'GoC_Axon_Xmin', 
        'goc-theta-basolateral-stdev' : 'GoC_Btheta_stdev', 
        'goc-theta-apical-stdev' : 'GoC_Atheta_stdev', 
        'goc-theta-apical-max' : 'GoC_Atheta_max', 
        'goc-axonsegs' : 'GoC_Axon_nseg', 
        'num-goc' : 'numGoC', 
        'goc-axonpts' : 'GoC_Axon_npts', 
        'z-extent' : 'GoCzrange', 
        'goc-axons' : 'numAxonGolgi', 
        'goc-theta-basolateral-max' : 'GoC_Btheta_max', 
        'pf-step' : 'PFstep', 
        'aa-step' : 'AAstep', 
        'x-extent' : 'GoCxrange', 
        'goc-apical-radius' : 'GoC_PhysApicalDendR', 
        'goc-axon-x-max' : 'GoC_Axon_Xmax', 
        'goc-basolateral-radius' : 'GoC_PhysBasolateralDendR', 
        'goc-goc-gap-zone' : 'GoCtoGoCgapzone', 
        'goc-apical-nseg' : 'GoC_Ad_nseg', 
        'goc-axon-y-min' : 'GoC_Axon_Ymin', 
        'goc-axon-z-max' : 'GoC_Axon_Zmax', 
        'goc-apical-dendheight' : 'GoC_PhysApicalDendH', 
        'goc-goc-zone' : 'GoCtoGoCzone', 
        'pf-goc-zone' : 'PFtoGoCzone', 
        'pf-length' : 'PFlength' }

        
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
        #https://stackoverflow.com/questions/12807539/how-do-you-convert-command-line-args-in-python-to-a-dictionary
        
    def check_output_prefix(self):
        '''checks if there is a specified output prefix.
        If not, will generate one from the timestampe'''
        if self.args.output == '':
            #! mkdir 'Res_{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
            self.args.output = 'Res_{:%Y-%m-%d_%H-%M-%S}/'.format(datetime.datetime.now())
            
    def read_in_config(self):
        '''checks if a config file has been specified, and if so, updates the args
        Warns if the file cannot be found, opened or properly read in (e.g. unkown parameter) '''
        if self.args.config == '':
            warnings.warn('Cannot find config file!')
            
    
    def p_verb (stat, *args):
        '''prints statement only if the print mode is on.
        Prints args in some smart ways'''
        pass
    
    def get_GC_Points():
        ''' 
        corresponds to GC-Points statement in brep.csm file.
        Read in the GC-Points provided in a file or render some randomly. (UniformRandomProcess)
        '''
        pass
        
    def get_GCT_Points():
        '''
        corresponds to the GCT-Points statement from the 
        '''
    
    def get_GOC_Points():
        '''
        get GOC Points from file or render them
        '''
        