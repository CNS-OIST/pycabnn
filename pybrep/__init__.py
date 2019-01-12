"""
pybrep.py

Finds distance-based connectivity between neurons with spatially extended
dendritic and axonal morphology, mainly developed for a physiologically detailed
network model of the cerebellar cortex.

Written by Ines Wichert and Sungho Hong, Computational Neuroscience Unit,
Okinawa Institute of Science and Technology

"""
from .connector import Connect_2D, Connect_3D

def create_population(celltype, h):
    from . import cell_population
    c = eval('cell_population.{}_pop'.format(celltype))
    return c(h)
