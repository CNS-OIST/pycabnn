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
gol_in = './example_simulation/coordinates_input/GoCcoordinates.sorted.dat'
#gol_in = './example_simulation/coordinates_input/subsampled/GoCcoordinates_64.dat'
gran_in = './example_simulation/coordinates_input/subsampled/GCcoordinates_64.dat'


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
gg.load_somata(gol_in)
#gg.gen_random_cell_loc(1995, 1500, 700, 200)
gg.add_dendrites()
gg.add_axon()
gg.save_axon_coords(global_prefix)
gg.save_dend_coords(global_prefix)
gg.save_somata (global_prefix, 'GoCcoordinates.sorted.dat')


cc = Connect_3D(gg.axon_q, gg.b_dend_q, 10, global_prefix+'GoCtoGoC_')
cc.connections_parallel(True)

