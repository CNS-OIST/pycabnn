#!/usr/bin/env python

from neuron import h
import pycabnn.cell_population as pop

soma_file = 'test_data/cell_position/MLIcoordinates.dat'
dend_file = 'test_data/MLI_dend_data_20211111.npz'

h.load_file('test_data/params/Parameters.hoc')
mlipop = pop.MLI_pop(h)
mlipop.load_somata(soma_file)
mlipop.add_dendrites()
mlipop.save_data(dend_file)
