import time
t1= time.time()
print ('Starting serial process...')

from BREPpy import *

np.random.seed (0)



#Parameters (might be read in from the command line some day...)
#Output path
global_prefix = './output/manyfiles_full/'
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
gg.save_dend_coords(global_prefix)
gg.save_somata (global_prefix, 'GoCcoordinates.sorted.dat')



t3= time.time()
print ('Golgi generation finished after', t3-t2)


#Set up Granule population including aa and pf
gp = Granule_pop(h)
gp.load_somata(gr_16)
gp.add_aa_endpoints_fixed()
gp.add_pf_endpoints()
gp.save_gct_points (global_prefix)
gp.save_somata (global_prefix, 'GCcoordinates.sorted.dat')

c_rad_aa = h.AAtoGoCzone
c_rad_pf = h.PFtoGoCzone


t4= time.time()
print ('Granule generation finished after', t4-t3)



###



import ipyparallel as ipp

rc = ipp.Client()
dv = rc[:]
lv = rc.load_balanced_view()

#cc_aa = Connect_2D(gg.qpts, gp.qpts_aa, c_rad_aa)
#kdt, q_pts, lax_c, lax_range, lin_in_tree = cc_aa.get_tree_setup()

qpts = gg.qpts
qpts_lin = gp.qpts_pf
c_rad = c_rad_pf
prefix = global_prefix + 'PFtoGoC'


#def connect_and_save (qpts, qpts_lin, c_rad, prefix):

st1 = time.time()

cc = Connect_2D(qpts, qpts_lin, c_rad)
kdt, q_pts, lax_c, lax_range, lin_in_tree = cc.get_tree_setup()

dv.block = True
with dv.sync_imports(): # note: import as does not work as only import part works, not assignment. Also, you will have to store the file in a directory reachable from the PYTHONPATH
    import parallel_util
#import numpy


con2d_dict = dict(
	kdt = kdt, 
	q_pts = q_pts, 
	c_rad = c_rad, 
	lin_axis = cc.lin_axis, 
	lin_in_tree = lin_in_tree,
	lin_is_src = cc.lin_is_src, 
	prefix = prefix, 
	pts = qpts,
	lpts = qpts_lin)

dv.push(con2d_dict)

lam_qpt = lambda ids: parallel_util.find_connections_2dpar (kdt, pts, lpts, c_rad, lin_axis, lin_in_tree, lin_is_src, ids, prefix)
dv.block = False


def get_id_array (len_id, id_sp, add_num = True):
	id_sp = int(id_sp)
	c = [np.arange(id_sp) + i * id_sp for i in range(int(np.ceil (len_id/id_sp)))]
	c[-1] = c[-1][c[-1]<len_id]
	if add_num:
		c= [[i, c[i]] for i in range(len(c))]
	return c

st2 = time.time()
print ('Copying done after', st2-st1)

spacing = np.ceil(len(q_pts)/len(rc.ids))
id_ar = get_id_array(len(q_pts), spacing)
bla = list(lv.map (lam_qpt, id_ar))

st3 = time.time()
print ('Calculation done after ', st3-st2)


#connect_and_save (qpts, qpts_lin, c_rad, prefix)
'''print (' ')
print ('STARTING PF')
connect_and_save(gg.qpts, gp.qpts_pf, c_rad_pf, global_prefix + 'PFtoGoC')
print (' ')
print ('STARTING AA')
connect_and_save(gg.qpts, gp.qpts_aa, c_rad_aa, global_prefix + 'AAtoGoC')
print (' ')
print ('Total time', st3-t1)
'''

'''idx, pre_res, pre_l_res = zip(*res_workers)

id_a = np.argsort(idx)
res = pre_res[id_a[0]]
l_res = pre_l_res[id_a[0]]
for n in id_a[1:]: 
	res.extend(pre_res[n])
	l_res.extend(pre_l_res[n])

st3 = time.time()
print ('Result obtained after ', st3-st2)

#cc_aa.save_results(res, l_res, global_prefix + 'AAtoGoC')

st4 = time.time()
print ('Saving finished after', st4-st3)'''



