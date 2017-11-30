

import time
t1= time.time()
print ('Starting sequential cess...')

from BREPpy import *


#Parameters (might be read in from the command line some day...)
#Output path
global_prefix = './output/initial_test/'
#config files
config_hoc = './input_data/Parameters.hoc'
config_pseudo_hoc = 'input_data/pseudo_hoc.pkl'
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



try:
    import neuron
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
gg.load_somata(go_16)
gg.add_dendrites()
gg.save_dend_coords('global_prefix')


t3= time.time()
print ('Golgi generation finished after', t3-t2)


#Set up Granule population including aa and pf
gp = Granule_pop(h)
gp.load_somata(gr_16)
gp.add_aa_endpoints_fixed()
gp.add_pf_endpoints()
gp.save_gct_points (global_prefix)


t4= time.time()
print ('Granule generation finished after', t4-t3)

#Build connector and obtain the connections
c_rad_aa = 15 # h.AAtoGoCzone
cc = Connect_2D(gg.qpts, gp.qpts_aa, c_rad_aa)
res_aa, l_res_aa = cc.find_connections()
cc.save_results (res_aa, l_res_aa, global_prefix+'AAtoGoC')

t5= time.time()
print ('AA connections found after', t5-t4)


c_rad_pf = 15 # h.PFtoGoCzone
cc = Connect_2D(gg.qpts, gp.qpts_pf, c_rad_pf)
res_pf, l_res_pf = cc.find_connections()
cc.save_results (res_pf, l_res_pf, global_prefix+'PFtoGoC')

t6= time.time()
print ('PF connections found after', t6-t5)



