


######### Please feed with data!!
#Parameters (might be read in from the command line some day...)
#Output path
output_path = './output/parallel12_full_4/'
# config files: if you work in an environment where you want
config_hoc = './input_data/Parameters.hoc'
config_pseudo_hoc = 'pseudo_hoc.pkl'
# Coordinate input files
gol_in = './input_data/GoCcoordinates.sorted.dat'
gran_in ='./input_data/GCcoordinates.sorted.dat'
#smaller dataset
#gol_in = './input_data/subsampled/GoCcoordinates_16.dat'
#gran_in = './input_data/subsampled/GCcoordinates_16.dat'


########  IMPORTS (probably you don't have to change anything here...)

import time
t1= time.time()
print ('Starting parallel process...')

from BREPpy import *
np.random.seed (0)


# Read in the config file (or the pseudo-hoc, see pseudo-hoc class to find out how it is generated)
try:
    import neuron
    h = neuron.hoc.HocObject()
    neuron.h.xopen(config_hoc)
    print ('Trying to read in hoc config object from ', config_hoc)
except:
    h = Pseudo_hoc(config_pseudo_hoc)
    print ('Trying to read in pseudo-hoc config object from ', config_pseudo_hoc)
finally:
	#Just pick a random variable and check whether it is read 
	assert hasattr(h, 'GoC_Atheta_min'), 'There might have been a problem reading in the parameters!'
	print ('Succesfully read in config file!')


t2= time.time()
print ('Import finished:', t2-t1)


########## POPULATION SETUP (to do here: change to random generation of cells)

# Set up the Golgi population, render dendrites
gg = Golgi_pop(h)
gg.load_somata(gol_in)
#gg.gen_random_cell_loc(1995, 1500, 700, 200)
gg.add_dendrites()
gg.save_dend_coords(output_path)
gg.save_somata (output_path, 'GoCcoordinates.sorted.dat')

t3= time.time()
print ('Golgi cell processing:', t3-t2)


#Set up Granule population including aa and pf
gp = Granule_pop(h)
gp.load_somata(gran_in)
#gp.gen_random_cell_loc(798000, 1500, 700, 200)
gp.add_aa_endpoints_fixed()
gp.add_pf_endpoints()
gp.save_gct_points (output_path)
gp.save_somata (output_path, 'GCcoordinates.sorted.dat')

t4= time.time()
print ('Granule cell processing:', t4-t3)
print (' ')



########### CONNECTIONS

#you might want to change the radii
c_rad_aa = h.AAtoGoCzone
c_rad_pf = h.PFtoGoCzone


cc = Connect_2D( gg.qpts, gp.qpts_aa, c_rad_aa, output_path+'AAtoGoC')
_ = cc.connections_parallel()


t5 = time.time()
print ('AA: Found and saved after', t5-t4)
print (' ')

cc = Connect_2D( gg.qpts,gp.qpts_pf, c_rad_pf, output_path+'PFtoGoC')
res_workers = cc.connections_parallel()

t6 = time.time()
print ('PF: Found and saved after', t6-t5)
print (' ')
