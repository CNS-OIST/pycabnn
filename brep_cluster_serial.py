
'''

import sys

sys.displayhook(stdout)
x = open('test_run_serial', 'w')
sys.displayhook(x)
sys.stdout = x
'''

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

c_rad_aa = h.AAtoGoCzone
c_rad_pf = h.PFtoGoCzone


t4= time.time()
print ('Granule generation finished after', t4-t3)



###

st1 = time.time()


import ipyparallel as ipp

rc = ipp.Client()
dv = rc[:]
print (len (rc.ids))
lv = rc.load_balanced_view()

cc_aa = Connect_2D(gg.qpts, gp.qpts_aa, c_rad_aa)
kdt_aa, q_pts_aa, lax_c, lax_range, lin_in_tree = cc_aa.get_tree_setup()

cc_pf = Connect_2D(gg.qpts, gp.qpts_pf, c_rad_pf)
kdt_pf, q_pts_pf, lax_c, lax_range, lin_in_tree = cc_pf.get_tree_setup()

dv.block = True
with dv.sync_imports(): # note: import as does not work as only import part works, not assignment. Also, you will have to store the file in a directory reachable from the PYTHONPATH
    import BREPpy
dv.push(dict(kdt_aa = kdt_aa, kdt_pf = kdt_pf, c_rad_aa = c_rad_aa, c_rad_pf = c_rad_pf))

lam_qpt_aa = lambda pt: BREPpy.pt_in_tr2(kdt_aa, pt, c_rad_aa)
lam_qpt_pf = lambda pt: BREPpy.pt_in_tr2(kdt_pf, pt, c_rad_pf)
#lam_qpt = lambda pt, c_rad: BREPpy.pt_in_tr2(kdt, pt, c_rad)
dv.block = False

st2 = time.time()
print ('copying done after', st2-st1)

res_aa = list(lv.map (lam_qpt_aa, q_pts_aa))
st3 = time.time()
print ('aa query ended after', st3-st2)


res_pf = list(lv.map (lam_qpt_pf, q_pts_pf))
st4 = time.time()
print ('pf query ended after', st4-st3)

