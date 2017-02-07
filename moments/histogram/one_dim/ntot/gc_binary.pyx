"""@docstring
@brief Manipulate binary systems sampled according to the order parameter N_tot
@author Nathan A. Mahynski
@date 02/07/2017
@filename gc_binary.pyx
"""

import gc_hist as gch
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

np.seterr(divide='raise', over='raise', invalid='raise', under='ignore') 

cdef inline double double_max(double a, double b): return a if a > b else b

def isopleth (object):
    """
    Class to compute the isopleths from a series of (mu1, dMu2) histograms
    
    """
    
    def __init__ (self, object histograms):
      """
      Instantiate the class.
    
      Parameters
      ----------
      histograms : array
        Array of hc_hist.histogram objects that define the space
      
      """
    
      self.clear()
      self.data['histograms'] = copy.deepcopy(histograms)
      
      # Check that all histograms at the same temperature and
    
    
    def clear (self):
      """
      Clear all information from the class.
    
      """
    
      self.data = {}
      self.meta = {}
    
    
