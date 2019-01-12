"""
pybrep.py

Finds distance-based connectivity between neurons with spatially extended
dendritic and axonal morphology, mainly developed for a physiologically detailed
network model of the cerebellar cortex.

Written by Ines Wichert and Sungho Hong, Computational Neuroscience Unit,
Okinawa Institute of Science and Technology

"""
import numpy as np
import csv
import warnings


####################################################################
## GENERAL UTILS PART                                             ##
####################################################################
from util import str_l

####################################################################
## POPULATION PART                                                ##
####################################################################

