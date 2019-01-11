'''
File to test the Connect_3D method on data that has been tested previously already.
Formerly called bprep4.py
'''

import time
t1= time.time()
print ('Starting parallel process...')

from BREPpy import *

np.random.seed (0)



#Parameters (might be read in from the command line some day...)
#Output path
global_prefix = './output_pybrep/spam/'
#config files
config_hoc = './input_data/Parameters.hoc'
#config_pseudo_hoc = 'input_data/pseudo_hoc.pkl'
config_pseudo_hoc = 'pseudo_hoc.pkl'
#input data (coordinates of the different cell somata)
p2 = './input_data/subsampled/'
go_ori = './input_data/GoCcoordinates.sorted.dat'
gr_ori ='./input_data/GCcoordinates.sorted.dat'
go_64 = p2+'GoCcoordinates_64.dat'
gr_64 = p2+'GCcoordinates_64.dat'
go_16 = p2+'GoCcoordinates_16.dat'
gr_16 = p2+'GCcoordinates_16.dat'
go_4 = p2+'GoCcoordinates_4.dat'
gr_4 = p2+'GCcoordinates_4.dat' 


gol_in = './example_simulation/coordinates_input/subsampled/GoCcoordinates_64.dat'
gran_in = './example_simulation/coordinates_input/subsampled/GCcoordinates_64.dat'


print ('Experiment: default parameter file, cell locations are generated randomly.')


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
print ('All imported after', t2-t1)


# Set up the Golgi population, render dendrites
gg = Golgi_pop(h)
gg.load_somata(go_64)
#gg.gen_random_cell_loc(1995, 1500, 700, 200)
gg.add_dendrites()
gg.save_dend_coords(global_prefix)
gg.save_somata (global_prefix, 'GoCcoordinates.sorted.dat')

t3= time.time()
print ('Golgi cell processing:', t3-t2)


#Set up Granule population including aa and pf
gp = Granule_pop(h)
gp.load_somata(gr_64)
#gp.gen_random_cell_loc(798000, 1500, 700, 200)
#gp.add_aa_endpoints_fixed()
#gp.add_pf_endpoints()
gp.add_3D_aa_and_pf()
gp.save_gct_points (global_prefix)
gp.save_somata (global_prefix, 'GCcoordinates.sorted.dat')

c_rad_aa = h.AAtoGoCzone
c_rad_pf = h.PFtoGoCzone

t2 = print_time_and_reset (t2, 'Golgi cell processing:')

#Test all 4 cases of different source and target populations, different populations in the tree
cc = Connect_3D(gp.qpts_aa,gg.qpts,  c_rad_aa, global_prefix+'AAtoGoC_3D_')
_ = cc.connections_parallel(True)
t2 = print_time_and_reset (t2)
cc = Connect_3D(gp.qpts_pf, gg.qpts, c_rad_pf, global_prefix+'PFtoGoC_3D_')
res_workers = cc.connections_parallel(True, False)
t2 = print_time_and_reset (t2)

cc = Connect_3D(gg.qpts, gp.qpts_aa, c_rad_aa, global_prefix+'AAtoGoC_3D_inv_')
_ = cc.connections_parallel(True)
t2 = print_time_and_reset (t2)
cc = Connect_3D(gg.qpts, gp.qpts_pf, c_rad_pf, global_prefix+'PFtoGoC_3D_inv_')
res_workers = cc.connections_parallel(True, True)
t2 = print_time_and_reset (t2)

cc = Connect_3D_parallel( gp.qpts_pf, gg.qpts, c_rad_pf, global_prefix+'PFtoGoC')
res_workers = cc.connections_parallel()

t6 = time.time()
print ('PF: Found and saved after', t6-t5)
print (' ')
