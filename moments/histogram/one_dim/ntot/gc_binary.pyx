"""@docstring
@brief Manipulate binary systems sampled according to the order parameter N_tot
@author Nathan A. Mahynski
@date 02/07/2017
@filename gc_binary.pyx
"""

import copy, cython, types
import numpy as np

cimport cython
cimport numpy as np

from numpy import ndarray
from numpy cimport ndarray
from cpython cimport bool
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport fabs

# Ignore underflows for phase property calculations.  This could be problematic if we are 
# interested in the properties of very unlikely phases, but since the primary goal here
# is to obtain phase behavior, we only care here about the properties of relatively likely
# phases.
np.seterr(divide='raise', over='raise', invalid='raise', under='ignore') 

cdef inline double double_max(double a, double b): return a if a > b else b
