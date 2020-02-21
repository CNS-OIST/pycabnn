import struct
import numpy as np
from tqdm.autonotebook import tqdm

type_length = {3: 4, # FLOAT
               4: 8, # #DOUBLE
               5: 4}  # INT

type_name = {3: 'float', # FLOAT
             4: 'double', # #DOUBLE
             5: 'int'}  # INT


def get_all_data_size(filename):
    n = 0
    ncell = 0
    ntype = None
    with open(filename, 'rb') as f:
        try:
            x = f.read(8) #read two integers
            on = True
        except:
            raise Exception('What!')

        while on:
            try:
                n1, ntype = struct.unpack('<ii', x) # read two integers
                n += n1 - 1 # subtract 1 from the cell index
                ncell += 1
                f.seek(n1*type_length[ntype], 1)
                x = f.read(8) # read two integers again
            except:
                f.close()
                on = False
        return (n, ncell, ntype)


def read_neuron_vectors(filename, data_size):
    ndata, ncell, ntype = data_size
    data = np.empty(ndata, dtype=type_name[ntype])
    cell = np.empty(ndata, dtype=int)

    count = 0
    mincell = ncell*100
    maxcell = 1
    with open(filename, 'rb') as f:
        for _ in tqdm(range(ncell)):
            x = f.read(8)
            n, ntype = struct.unpack('<ii', x)
            x = f.read(n*type_length[ntype])
            if n>1:
                st = np.fromstring(x, dtype=type_name[ntype])
                data[count:(count+n-1)] = st[1:]
                cell[count:(count+n-1)] = st[0]
                count = count + n-1
            else:
                st = np.fromstring(x, dtype=type_name[ntype])
            if mincell>st[0]:
                mincell = st[0]
            if maxcell<st[0]:
                maxcell = st[0]
        # print mincell, maxcell
    return (data, cell, int(mincell)-1)
