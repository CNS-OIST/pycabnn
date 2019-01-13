#!/usr/bin/env python
"""run_pybrep.py

Jobs = AAtoGoC, PFtoGoC, GoCtoGoC, GoCtoGoCgap

Usage:
  run_pybrep.py (-i PATH) (-o PATH) (-p PATH) [--parallel] (all | <jobs>...)
  run_pybrep.py (-h | --help)
  run_pybrep.py --version

Options:
  -h --help                            Show this screen.
  --version                            Show version.
  -i PATH, --input_path=<input_path>   Input path.
  -o PATH, --output_path=<output_path> Output path.
  -p PATH, --param_path=<param_dir>    Params path.
  --parallel                           Run things in parallel (via ipyparallel)

"""

from pathlib import Path
import pybrep as brp
import time
from docopt import docopt
from neuron import h


def load_input_data(args):
    data = {}
    for arg in args:
        if 'path' in arg:
            data[arg.replace('--', '')] = Path(args[arg]).resolve()

    data['config_hoc'] = data['param_path'] / 'Parameters.hoc'

    t1 = time.time()
    print('Starting parallel process...')

    # Read in the config file(or the pseudo-hoc, see pseudo-hoc class to find out how it is generated)
    try:
        from neuron import h
        config_hoc = str(data['config_hoc'])
        h.xopen(config_hoc)
        print('Trying to read in hoc config object from ', config_hoc)
    except ModuleNotFoundError:
        # TODO: have to find a good way to deal with this
        config_pseudo_hoc = Path.cwd() / 'pseudo_hoc.pkl'
        h = brp.Pseudo_hoc(config_pseudo_hoc)
        print('Trying to read in pseudo-hoc config object from ', config_pseudo_hoc)
    finally:
        #Just pick a random variable and check whether it is read
        assert hasattr(h, 'GoC_Atheta_min'), 'There might have been a problem reading in the parameters!'
        print('Succesfully read in config file!')

    t2 = time.time()
    print('Import finished:', t2-t1)
    data['t'] = t2
    return data


def make_population(data):
    output_path = data['output_path']

    # Set up the Golgi population, render dendrites
    gol_in = data['input_path'] / 'GoCcoordinates.dat'
    gg = brp.create_population('Golgi', h)
    gg.load_somata(gol_in)
    # # gg.gen_random_cell_loc(1995, 1500, 700, 200)
    gg.add_dendrites()
    gg.save_dend_coords(output_path)
    gg.add_axon()
    gg.save_somata(output_path, 'GoCcoordinates.sorted.dat')

    t2 = data['t']
    t3 = time.time()
    print('Golgi cell processing:', t3-t2)

    #Set up Granule population including aa and pf

    gran_in = data['input_path'] / 'GCcoordinates.dat'
    gp = brp.create_population('Granule', h)
    gp.load_somata(gran_in)
    gp.add_aa_endpoints_fixed()
    gp.add_pf_endpoints()
    gp.save_gct_points(output_path)
    gp.save_somata(output_path, 'GCcoordinates.sorted.dat')

    t4 = time.time()
    print('Granule cell processing:', t4-t3)
    print(' ')

    data['t'] = t4
    data['pops'] = {'grc': gp, 'goc': gg}

    return data


def run_AAtoGoC(data):
    # you might want to change the radii
    c_rad_aa = h.AAtoGoCzone/1.73
    print("R for AA: {}".format(c_rad_aa))

    gp, gg = data['pops']['grc'], data['pops']['goc']
    cc = brp.Connect_2D(gp.qpts_aa, gg.qpts, c_rad_aa)
    # TODO: adapt the following from command line options
    cc.connections_parallel(deparallelize=False, nblocks=120, debug=False)
    cc.save_result( data['output_path'] / 'AAtoGoC')

    t5 = time.time()
    t4 = data['t']
    print('AA: Found and saved after', t5-t4)
    print(' ')
    data['t'] = t5
    return data


def run_PFtoGoC(data):
    c_rad_pf = h.PFtoGoCzone/1.113
    print("R for PF: {}".format(c_rad_pf))

    gp, gg = data['pops']['grc'], data['pops']['goc']
    cc = brp.Connect_2D(gp.qpts_pf, gg.qpts, c_rad_pf)
    cc.connections_parallel(deparallelize=False, nblocks=120, debug=False)
    cc.save_result(data['output_path'] / 'PFtoGoC')

    t6 = time.time()
    t5 = data['t']
    print('PF: Found and saved after', t6-t5)
    print(' ')
    data['t'] = t6
    return data


def run_GoCtoGoC(data):
    import numpy as np
    from tqdm import tqdm

    ##### TODO: need to be reimplemented by pybrep
    from sklearn.neighbors import KDTree

    gg = data['pops']['goc']
    dist = []
    src = []
    tgt = []

    for i in tqdm(range(gg.n_cell)):
        axon_coord1 = gg.axon[i]
        tree = KDTree(axon_coord1)
        for j in range(gg.n_cell):
            if i != j:
                ii, di = tree.query_radius(np.expand_dims(gg.som[j], axis=0),
                                           r=h.GoCtoGoCzone,
                                           return_distance=True)
                if ii[0].size > 0:
                    temp = di[0].argmin()
                    ii, di = ii[0][temp], di[0][temp]
                    axon_len = np.linalg.norm(axon_coord1[ii]-gg.som[i])
                    src.append(i)
                    tgt.append(j)
                    dist.append(axon_len + di) # putative path length along the axon

    output_path = data['output_path']

    np.savetxt(output_path / 'GoCtoGoCsources.dat', src, fmt='%d')
    np.savetxt(output_path / 'GoCtoGoCtargets.dat', tgt, fmt='%d')
    np.savetxt(output_path / 'GoCtoGoCdistances.dat', dist)

    t2 = time.time()
    t1 = data['t']
    print('GoCtoGoC: Found and saved after', t2-t1)
    data['t'] = t2
    return data


def run_GoCtoGoCgap(data):
    import numpy as np
    from tqdm import tqdm

    gg = data['pops']['goc']
    dist = []
    src = []
    tgt = []

    for i in tqdm(range(gg.n_cell)):
        for j in range(gg.n_cell):
            if i != j:
                di = np.linalg.norm(gg.som[j]-gg.som[i])
                if di < h.GoCtoGoCzone:
                    src.append(i)
                    tgt.append(j)
                    dist.append(di)

    output_path = data['output_path']
    np.savetxt(output_path / 'GoCtoGoCgapsources.dat', src, fmt='%d')
    np.savetxt(output_path / 'GoCtoGoCgaptargets.dat', tgt, fmt='%d')
    np.savetxt(output_path / 'GoCtoGoCgapdistances.dat', dist)

    t2 = time.time()
    t1 = data['t']
    print('GoCtoGoCgap: Found and saved after', t2-t1)
    data['t'] = t2
    return data


def main(args):
    data = load_input_data(args)
    data = make_population(data)

    valid_job_list = ['AAtoGoC', 'PFtoGoC', 'GoCtoGoC', 'GoCtoGoCgap']
    if args['all']:
        args['<jobs>'] = valid_job_list
    for j in args['<jobs>']:
        if j not in valid_job_list:
            raise RuntimeError('Job {} is not valid, not in {}'.format(j, valid_job_list))
        else:
            data = eval('run_'+j)(data)


if __name__ == '__main__':
    args = docopt(__doc__, version='0.7dev')
    main(args)
