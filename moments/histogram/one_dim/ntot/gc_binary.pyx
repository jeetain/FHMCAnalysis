"""@docstring
@brief Manipulate binary systems sampled according to the order parameter N_tot
@author Nathan A. Mahynski
@date 02/07/2017
@filename gc_binary.pyx
"""

import gc_hist as gch
import copy, cython, types, operator
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

		cdef double dmu2

        if (not isinstance(histogram, list)): raise Exception ('Expects a vector of histograms to construct isopleths')
        self.clear()

        # Check that all histograms at the same temperature and 2 species
        b = None
        for h in histograms:
            if (b is not None):
                b = h.data['curr_beta']
            else:
                if (fabs(b-h.data['curr_beta']) > self.data['tol']): raise Exception ('Temperature mismatch in isopleth generation')
                if (h.data['nspec'] != 2): raise Exception ('Component mismatch in isopleth generation')

        # Sort the histograms from min to max dmu2
		dummy = {}
		for h in histograms:
			if (len(h.data['curr_mu']) != 2): raise Exception ('Only expects 2 chemical potentials, one for each component, cannot construct isopleth')
			dmu2 = float(h.data['curr_mu'][1] - h.data['curr_mu'][0])
			dummy[dmu2] = copy.deepcopy(h)

		self.data['histograms'] = sorted(x.items(), key=operator.itemgetter(0)) # dict of {dmu2:histogram}

    def clear (self):
        """
        Clears all non-default information from the class.

        """

        self.data = {}
        self.data['tol'] = 1.0e-9

    def make_grid (self, mu1_bounds, mu2_bounds, delta):
        """
        Compute the discretized 2D (mu1, mu2) isopleth surface.

        Parameters
        ----------
        mu1_bounds : tuple
            min, max of mu_1 to consider
        mu2_bounds : tuple
            min, max of mu_2 to consider
        delta : tuple
            Width of mu bins to use in each (mu1, mu2) dimension on a discrete grid

        Returns
        -------
        grid_x1 : ndarray
            2D array of x1
        grid_mu : ndarray
            3D array of (mu1, mu2) at each "pixel"

        """

        cdef int i, j

        if (len(mu1_bounds) != 2): raise Exception ('mu1_bound error in constructing isopleths')
        if (len(mu2_bounds) != 2): raise Exception ('mu2_bound error in constructing isopleths')
        if (len(delta) != 2): raise Exception ('delta error in constructing isopleths')
        if (mu1_bounds[1] <= mu1_bounds[0]): raise Exception ('mu1_bound error in constructing isopleths')
        if (mu2_bounds[1] <= mu2_bounds[0]): raise Exception ('mu2_bound error in constructing isopleths')
        if (delta[0] <= 0): raise Exception ('delta error in constructing isopleths')
        if (delta[1] <= 0): raise Exception ('delta error in constructing isopleths')

        # Compute x1 at each point in the grid
        cdef int nx = np.ceil((mu1_bounds[1]-mu1_bounds[0])/delta[0])+1
        cdef int ny = np.ceil((mu2_bounds[1]-mu2_bounds[0])/delta[1])+1

        grid_x1 = np.zeros((nx, ny), dtype=np.float64)
        grid_mu = np.zeros((nx, ny, 2), dtype=np.float64)

        for i in range(0, nx):
            for j in range(0, ny):
                mu1 = mu1_bounds[0] + delta[0]*i
                mu2 = mu2_bounds[0] + delta[1]*j
                grid_mu[i,j,:] = [mu1, mu2]

                # Identify "bounding" dMu2's

                # Reweight to target mu1

                # Extrapolate each to target dMu2 for each to reach this (mu1, mu2)

                # "Mix" histograms based on distance that had to be reweighted

                # Compute properties of "mixed" histograms

        return grid_x1, grid_mu

    def get_iso (self, x1):
        """
        Trace out the isopleth from the discretized grid of (mu1, mu2)

        Parameters
        ----------
        x1 : double
            Target mole fraction of species 1

        Returns
        -------
        mu_vals : array
            (mu1, mu2) values that trace out the isopleth

        """

        return mu_vals
