"""
data_io

- data_io.OutputReader is a class for reading the output simulated data.
- data_io.MFHandler is a class for reading and writing the input to the network simulation (mossy fiber information).
- repack_dict converts the spike time data in a dictionary form to a table in a pandas.DataFrame format.

"""

import os
import pandas as pd
import numpy as np

from . import geometry
from . import nrnvector
from pathlib import Path


def rpath(f):
    def wrapped(self=None, filename=None):
        return f(self, self.root.joinpath(filename))
    return wrapped

def attach_coords(data, coords):
    cells, cref = data['cell'].values, coords.values
    dims = cref.shape[1]

    @np.vectorize
    def f(cell, d):
        return cref[cell-1, d]

    xyz = pd.DataFrame(np.array([f(cells, d) for d in range(dims)]).T)
    xyz.columns = ['x', 'y', 'z'][:dims]
    return pd.concat([data, xyz], axis=1)


class OutputReader(object):
    def __init__(self, root):
        if not os.path.exists(root):
            raise Exception('No data found at ' + root)
        else:
            self.root = Path(root).resolve()

            desc_file = [f for f in self.root.glob('*DESCRIPTION.txt') if f.is_file()]
            if len(desc_file)>0:
                desc_file, = desc_file
                with open(desc_file) as f:
                    self.desc = f.read()
                    print(self.desc)
            else:
                self.desc = ""

            caseset, = [d for d in self.root.glob(pattern='set*') if d.is_dir()]
            self.caseset = caseset.name
            desc_set = ('Used set = %s' % self.caseset)
            print(desc_set)
            self.desc += desc_set

    @rpath
    def read_neuron_vectors(self, filename):
        """ read_neuron_vectors """
        def _read_neuron_vectors(filename):
            dsize = nrnvector.get_all_data_size(filename)
            print('Found ', dsize[0], 'spikes from', dsize[1], 'cells.')
            spiketime, cell, _ = nrnvector.read_neuron_vectors(filename, dsize)
            return pd.DataFrame({'time':spiketime, 'cell':cell})

        return _read_neuron_vectors(filename)

    @rpath
    def read_sorted_coords(self, filename):
        """ read_sorted_coords """
        def _read_sorted_coords(filename):
            coords = pd.read_table(filename, names='xyz', sep=' ')
            coords.index = coords.index+1
            return coords
        return _read_sorted_coords(filename)

    @rpath
    def read_MF_coords(self, filename):
        def _read_out_MFcoords(filename):
            coords = []
            with open(filename) as f:
                for l in f.readlines():
                    xy = map(float, l.strip('\n').split('\t')[:2])
                    if len(xy)==2:
                        coords.append(xy)
            return pd.DataFrame(np.array(coords), columns=['x', 'y'])
        return _read_out_MFcoords(filename)

    def read_coords(self, cell):
        celldict = {'grc': 'GCcoordinates.sorted.dat',
                    'goc': 'GoCcoordinates.sorted.dat',
                    'mf' : 'MFcoordinates.dat'}
        if cell not in celldict:
            raise IOError(cell + ' not found in ' + str(celldict.keys()))
        else:
            fcoord = celldict[cell]

            if 'sorted' in fcoord:
                fread_coord = self.read_sorted_coords
            else:
                if cell=='mf':
                    fread_coord = self.read_MF_coords
                else:
                    raise RuntimeError('Do not know how to read the coordinates for ' + cell)

            coords = fread_coord(fcoord)
        return coords

    def read_spike_data(self, cell, with_coords=True):
        celldict = {'grc': 'Gspiketime.bin',
                    'goc': 'GoCspiketime.bin',
                    'mf': 'MFspiketime.bin'}
        if cell not in celldict:
            raise Exception(cell + ' not found in ' + str(celldict.keys()))
        else:
            fspike = celldict[cell]
            spikedata = self.read_neuron_vectors(fspike)

        if with_coords:
            spikedata = attach_coords(spikedata, self.read_coords(cell))

        return spikedata

    def read_connectivity(self, pre, post, with_delay=False, with_coords=True):
        if pre=='grc':
            raise RuntimeError('You need to specify the pathway (aa or pf)')

        conndict = {'mf->grc' : 'MFtoGC',
                    'mf->goc' : 'MFtoGoC',
                    'aa->goc' : 'AxontoGoC',
                    'pf->goc' : 'PFtoGoC',
                    'goc->grc': 'GoCtoGC'}

        conn = pre+'->'+post

        def _read_neuron_vectors(filename):
            dsize = nrnvector.get_all_data_size(filename)
            print('Found ', dsize[0], 'connections to', dsize[1], 'cells.')
            precs, postcs, offset = nrnvector.read_neuron_vectors(filename, dsize)
            postcs = postcs - offset
            return pd.DataFrame({'pre':precs.astype(int), 'cell':postcs})

        if conn not in conndict:
            raise Exception(conn + ' not found in ' + str(conndict.keys()))
        else:
            fconn = conndict[conn] + '.bin'
            conndata = _read_neuron_vectors(self.root.joinpath(fconn))

        if with_coords:
            conndata = attach_coords(conndata, self.read_coords(post))

        return conndata


class MFHandler(object):
    def __init__(self, root):
        """
        `mf = MFHandler(path)` creates an MFHandle object to read and write data at _path_.
        """
        if not os.path.exists(root):
            raise Exception('No data found at ' + root)
        else:
            self.root = Path(root).abspath()

    def read_spike_data(self, with_coords=True):
        "xy = MFHandler.read_spike_data() read MF data file based on the number type (int if it's length file; float if it's spiketrain file)"
        def _read_datasp(filename):
            time = []
            with open(filename) as f:
                for l in f.readlines():
                    time.append([float(x) for x in l.strip('\n').split('\t') if len(x)>0])
            return time

        datasp = _read_datasp(self.root.joinpath('datasp.dat'))

        def _read_l(filename):
            ldata = []
            with open(filename) as f:
                for l in f.readlines():
                    temp = [x for x in l.strip('\n').split('\t') if len(x)>0]
                    if len(temp)>0:
                        if len(temp)==1:
                            ldata.append(int(temp[0]))
                        else:
                            raise Exception('Something is wrong.')

            return ldata

        l = _read_l(self.root.joinpath('l.dat'))

        def _check_length(datasp, l):
            checksum = np.sum(int(len(sp1)==l1) for sp1, l1 in zip(datasp, l))
            if checksum != len(l):
                raise Exception('l.dat and datasp.dat are inconsistent.')
            else:
                print('Successfully read MF spike time data.')

        _check_length(datasp, l)

        def _read_active_cells(filename):
            with open(filename) as f:
                x = [int(x) for x in f.read().strip('\n').split('\t') if len(x)>0]
            return x

        active_cells = _read_active_cells(self.root.joinpath('activeMfibres1.dat'))

        nspikes = np.sum(l)
        time = np.empty((nspikes,), dtype=float)
        cells = np.empty((nspikes,), dtype=int)

        count = 0
        for i, data in enumerate(datasp):
            time[count:(count+len(data))] = data
            cells[count:(count+len(data))] = active_cells[i]
            count = count+len(data)

        df = pd.DataFrame({'time': time, 'cell':cells})

        if with_coords:
            df = attach_coords(df, self.read_coordinates())

        return df

    def read_coordinates(self):
        """xy = MFHandler.read_coordinates() read the coordinates of mossy fibers"""
        def _read_set_MFcoordinates(filename):
            import csv
            data = []
            results = []
            for row in csv.reader(open(filename), delimiter=' '):
                data.append(row)
            for d in data:
                d = [float(i) for i in d]
                results.append(d)
            xy = pd.DataFrame(np.array(results), columns=['x', 'y'])
            return xy

        return _read_set_MFcoordinates(self.root.joinpath('MFCr.dat'))

    def read_tstop(self):
        import re

        re1='.*?(\\d+)'
        rg = re.compile(re1,re.IGNORECASE|re.DOTALL)

        with open(self.root.joinpath('Parameters.hoc'), 'r') as f:
            line = ''
            while 'stoptime' not in line:
                line = f.readline()

        m = rg.search(line)
        if m:
            int1=m.group(1)
        else:
            raise RuntimeError("Cannot extract stoptime.")
        return float(int1)


def repack_dict(d, coords=None):
    nspikes = np.sum(len(d[i]) for i in d)
    count = 0
    time = np.empty((nspikes,), dtype=float)
    cells = np.empty((nspikes,), dtype=int)
    for i in d:
        data = d[i]
        time[count:(count+len(data))] = data
        cells[count:(count+len(data))] = i
        count = count+len(data)
    df = pd.DataFrame({'time': time, 'cell': cells})
    if coords is not None:
        df = attach_coords(df, coords)

    return df


def save_spikes_mat(root, savepath, extra_info=''):
    """save_spikes_mat(output root, where to save, extra info str)"""
    cells = ['grc', 'goc', 'mf']
    out = OutputReader(root)
    data = dict((c, out.read_spike_data(c)) for c in cells)

    def add_location_data(d, df):
        if 'z' in df.keys():
            d['xyz'] = np.vstack([df[x] for x in 'xyz'])
        else:
            d['xy'] = np.vstack([df[x] for x in 'xy'])
        return d

    def spikedf_to_dict(df):
        d = {}
        d['cell'] = df.cell.values
        d['time'] = df.time.values
        d = add_location_data(d, df)
        return d

    def reformat_dict(data, fformat):
        outd = {}
        for k in data:
            outd[k] = fformat(data[k])
        return outd

    temp = reformat_dict(data, spikedf_to_dict)
    temp['description'] = out.desc
    _, dataid = out.root.splitext()
    dataid = dataid[1:]

    temp['stimset'] = str(out.caseset)

    locdata = dict((c+'xy', out.read_coords(c)) for c in cells)
    for k in locdata:
        temp[k] = add_location_data({}, locdata[k])

    from scipy.io import savemat
    p = Path(savepath).joinpath('spiketime_'+dataid+'_'+out.caseset+'_'+extra_info+'.mat')
    savemat(p, temp, do_compression=True)
    return p
