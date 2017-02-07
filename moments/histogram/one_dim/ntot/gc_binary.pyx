"""@docstring
@brief Manipulate binary systems sampled according to the order parameter N_tot
@author Nathan A. Mahynski
@date 02/07/2017
@filename gc_binary.pyx
"""

import gc_hist as gch
import copy, cython, types, operator, bisect
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

	def __init__ (self, object histograms, double beta_target, int order=2):
		"""
		Instantiate the class.

		Parameters
		----------
		histograms : array
			Array of hc_hist.histogram objects that define the space
		beta_target : double
			Desired value of beta = 1/kT to generate isopleths at
		order : int
			Order of extrapolation to use (default=2)

		"""

		cdef double dmu2

		if (not isinstance(histograms, list)): raise Exception ('Expects a vector of histograms to construct isopleths')
		for h in histograms:
			if (not isinstance(h, gch.histogram)): raise Exception ('Expects a vector of histograms to construct isopleths')
		if (beta_target <= 0): raise Exception ('Illegal beta, cannot construct isopleths')
		if (order < 1 or order > 2): raise Exception ('Illegal order, cannot construct isopleths')

		self.meta = {}
		self.meta['beta'] = beta_target
		self.meta['tol'] = 1.0e-9
		self.meta['order'] = order
		self.meta['cutoff'] = 10.0
		self.clear()

		# Check that all histograms have 2 species
		for h in histograms:
			if (h.data['nspec'] != 2): raise Exception ('Component mismatch in isopleth generation')

		# Sort the histograms from min to max dmu2
		dummy = {}
		for h in histograms:
			if (len(h.data['curr_mu']) != 2): raise Exception ('Only expects 2 chemical potentials, one for each component, cannot construct isopleth')
			dmu2 = float(h.data['curr_mu'][1] - h.data['curr_mu'][0])
			dummy[dmu2] = h
		dummy_sorted = sorted(dummy.items(), key=operator.itemgetter(0)) # dict of {dmu2:histogram}

		self.data['dmu2'] = [x[0] for x in dummy_sorted]
		self.data['histograms'] = [copy.deepcopy(x[1]) for x in dummy_sorted]

	def clear (self):
		"""
		Clears all information from the class.  Leaves metadata intact.

		"""

		self.data = {}

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
		cdef double dmu2, mu1, mu2

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
				dmu2 = mu2 - mu1
				grid_mu[i,j,:] = [mu1, mu2]

				# Identify "bounding" dmu2's
				left = bisect.bisect_left(self.data['dmu2'], dmu2)
				right = bisect.bisect_right(self.data['dmu2'], dmu2)

				if (left == right and left == 0):
					# below the lowest bounds, extrapolate just the lowest bound
					h_l = self.data['histograms'][left]
					try:
						h_l.reweight(mu1)
						h_l = h_l.temp_dmu_extrap(self.meta['beta'], np.array([dmu2], dtype=np.float64), self.meta['order'], self.meta['cutoff'], False, True, False)
						h_l.thermo()
						if (not h_l.is_safe()):
							raise Exception ('extrapolated ln(PI) in histogram is not safe to use')
						else:
							# find most stable phase and extract properties
							return

					except Exception as e:
						raise Exception ('Error at (mu1,mu2) = ('+str(mu1)+','+str(mu2)+') : '+str(e))
				elif (left == right and left == len(self.data['dmu2'])):
					# above top bound, extrapolate just the upper bound
					h_r = self.data['histograms'][right-1]
					try:
						h_r.reweight(mu1)
						h_r = h_r.temp_dmu_extrap(self.meta['beta'], np.array([dmu2], dtype=np.float64), self.meta['order'], self.meta['cutoff'], False, True, False)
						h_r.thermo()
						if (not h_r.is_safe()):
							raise Exception ('extrapolated ln(PI) in histogram is not safe to use')
						else:
							# find most stable phase and extract properties
							return

					except Exception as e:
						raise Exception ('Error at (mu1,mu2) = ('+str(mu1)+','+str(mu2)+') : '+str(e))
				elif (fabs(dmu2 - self.data['dmu2'][left]) < self.meta['tol']):
					# exactly equal to the dmu2 value at left, just use this one (no extrapolation necessary)
					h_l = self.data['histograms'][left]
					try:
						h_l.reweight(mu1)
						h_l = h_l.temp_dmu_extrap(self.meta['beta'], np.array([dmu2], dtype=np.float64), self.meta['order'], self.meta['cutoff'], False, True, False)
						h_l.thermo()
						if (not h_l.is_safe()):
							raise Exception ('extrapolated ln(PI) in histogram is not safe to use')
						else:
							# find most stable phase and extract properties
							return

					except Exception as e:
						raise Exception ('Error at (mu1,mu2) = ('+str(mu1)+','+str(mu2)+') : '+str(e))
				else:
					# in between two measured dmu2 values
					h_l = self.data['histograms'][left]
					try:
						h_l.reweight(mu1)
						h_l = h_l.temp_dmu_extrap(self.meta['beta'], np.array([dmu2], dtype=np.float64), self.meta['order'], self.meta['cutoff'], False, True, False)
					except Exception as e:
						raise Exception ('Error at (mu1,mu2) = ('+str(mu1)+','+str(mu2)+') : '+str(e))

					h_r = self.data['histograms'][right]
					try:
						h_r.reweight(mu1)
						h_r = h_r.temp_dmu_extrap(self.meta['beta'], np.array([dmu2], dtype=np.float64), self.meta['order'], self.meta['cutoff'], False, True, False)
					except Exception as e:
						raise Exception ('Error at (mu1,mu2) = ('+str(mu1)+','+str(mu2)+') : '+str(e))

					# Mix these histograms
					d1 = fabs(self.data['dmu2'][left] - dmu2)
					d2 = fabs(self.data['dmu2'][right] - dmu2)

					"""rel_weight =
					h_m = h_l.mix(h_r, rel_weight)

					try:
						h_m.thermo()
						if (not h_m.is_safe()):
							raise Exception ('extrapolated ln(PI) in histogram is not safe to use')
					except Exception as e:
						raise Exception ('Error at (mu1,mu2) = ('+str(mu1)+','+str(mu2)+') : '+str(e))
					else:
						# find most stable phase and extract properties
						return"""


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
			array of tuples (mu1, mu2) that trace out the isopleth

		"""

		mu_vals = []

		return mu_vals
