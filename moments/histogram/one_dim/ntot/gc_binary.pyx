"""@docstring
@brief Manipulate binary systems sampled according to the order parameter N_tot
@author Nathan A. Mahynski
@date 02/07/2017
@filename gc_binary.pyx
"""

import gc_hist as gch
import numpy as np
import matplotlib.pyplot as plt
import copy, cython, types, operator, bisect

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

def _get_most_stable_phase (hist):
	"""
	Return the index of the most stable phase after thermo() calculation.

	Parameters
	----------
	hist : gc_hist.histogram
		Histogram that thermodynamics was computed for

	Returns
	-------
	int
		Index of most stable phase in hist

	"""

	cdef int most_stable_phase = 0

	free_energy = {}
	for phase in hist.data['thermo']:
		free_energy[phase] = phase['F.E./kT']
	free_energy_sorted = sorted(free_energy.items(), key=operator.itemgetter(1))
	most_stable_phase = free_energy_sorted[0][0]

	return most_stable_phase

class isopleth (object):
	"""
	Class to compute the isopleths from a series of (mu1, dMu2) histograms

	"""

	def __init__ (self, object histograms, double beta_target, int order=2):
		"""
		Instantiate the class.

		Parameters
		----------
		histograms : array
			Array of gc_hist.histogram objects that define the space
		beta_target : double
			Desired value of beta = 1/kT to generate isopleths at
		order : int
			Order of extrapolation to use (default=2)

		"""

		cdef double dmu2

		if (not isinstance(histograms, (list, np.ndarray))): raise Exception ('Expects an array of histograms to construct isopleths')
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
		Uses "linear" mixing to combine extrapolated histograms.

		Parameters
		----------
		mu1_bounds : tuple
			min, max of mu_1 to consider
		mu2_bounds : tuple
			min, max of mu_2 to consider
		delta : tuple or array
			Width of mu bins to use in each (mu1, mu2) dimension on a discrete grid

		Returns
		-------
		grid_x1 : ndarray
			2D array of x1 (< 0 where thermodynamics could not be calculated)
		grid_mu : ndarray
			3D array of (mu1, mu2) at each "pixel"

		"""

		cdef int i, j
		cdef double dmu2, mu1, mu2

		if (len(mu1_bounds) != 2): raise Exception ('mu1_bound error in constructing isopleths')
		if (len(mu2_bounds) != 2): raise Exception ('mu2_bound error in constructing isopleths')
		if (not isinstance(delta, (list, np.ndarray))): raise Exception ('Expects an array of delta mu values to construct isopleths')
		if (len(delta) != 2): raise Exception ('delta error in constructing isopleths')
		if (mu1_bounds[1] <= mu1_bounds[0]): raise Exception ('mu1_bound error in constructing isopleths')
		if (mu2_bounds[1] <= mu2_bounds[0]): raise Exception ('mu2_bound error in constructing isopleths')
		if (delta[0] <= 0): raise Exception ('delta error in constructing isopleths')
		if (delta[1] <= 0): raise Exception ('delta error in constructing isopleths')

		# Compute x1 at each point in the grid
		cdef int nx = np.ceil((mu1_bounds[1]-mu1_bounds[0])/delta[0])+1
		cdef int ny = np.ceil((mu2_bounds[1]-mu2_bounds[0])/delta[1])+1

		grid_x1 = np.zeros((nx, ny), dtype=np.float64) - 1.0
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
							most_stable_phase = _get_most_stable_phase(h_l)
							grid_x1[i,j] = h_l.data['thermo'][most_stable_phase]['x1']
					except Exception as e:
						print 'Error at (mu1,mu2) = ('+str(mu1)+','+str(mu2)+') : '+str(e)+', continuing on...'
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
							most_stable_phase = _get_most_stable_phase(h_r)
							grid_x1[i,j] = h_r.data['thermo'][most_stable_phase]['x1']
					except Exception as e:
						print 'Error at (mu1,mu2) = ('+str(mu1)+','+str(mu2)+') : '+str(e)+', continuing on...'
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
							most_stable_phase = _get_most_stable_phase(h_l)
							grid_x1[i,j] = h_l.data['thermo'][most_stable_phase]['x1']
					except Exception as e:
						print 'Error at (mu1,mu2) = ('+str(mu1)+','+str(mu2)+') : '+str(e)+', continuing on...'
				else:
					# in between two measured dmu2 values
					h_l = self.data['histograms'][left]
					h_r = self.data['histograms'][right]
					try:
						h_l.reweight(mu1)
						h_l = h_l.temp_dmu_extrap(self.meta['beta'], np.array([dmu2], dtype=np.float64), self.meta['order'], self.meta['cutoff'], False, True, False)
						h_r.reweight(mu1)
						h_r = h_r.temp_dmu_extrap(self.meta['beta'], np.array([dmu2], dtype=np.float64), self.meta['order'], self.meta['cutoff'], False, True, False)
					except Exception as e:
						print 'Error at (mu1,mu2) = ('+str(mu1)+','+str(mu2)+') : '+str(e)+', continuing on...'
					else:
						# "linearly" mix these histograms
						dl = fabs(self.data['dmu2'][left] - dmu2)
						dr = fabs(self.data['dmu2'][right] - dmu2)

						h_m = h_l.mix(h_r, [dl/(dl+dr), dr/(dl+dr)])
						try:
							h_m.thermo()
							if (not h_m.is_safe()):
								raise Exception ('extrapolated ln(PI) in histogram is not safe to use')
						except Exception as e:
							print 'Error at (mu1,mu2) = ('+str(mu1)+','+str(mu2)+') : '+str(e)+', continuing on...'
						else:
							# find most stable phase and extract properties
							most_stable_phase = _get_most_stable_phase(h_m)
							grid_x1[i,j] = h_m.data['thermo'][most_stable_phase]['x1']

		return grid_x1, grid_mu

	def get_iso (self, double x1, grid_x1, grid_mu):
		"""
		Trace out the isopleth from the discretized grid of (mu1, mu2)

		Parameters
		----------
		x1 : double
			Target mole fraction of species 1
		grid_x1 : ndarray
			2D array of x1
		grid_mu : ndarray
			3D array of (mu1, mu2) at each "pixel"

		Returns
		-------
		mu_vals : array
			array of tuples (mu1, mu2) that trace out the isopleth

		"""

		cs = plt.contour(grid_mu[:,:,0], grid_mu[:,:,1], grid_x1, [x1])
		p = cs.collections[0].get_paths()[0]
		v = p.vertices
		mu_vals = zip(v[:,0], v[:,1])

		return mu_vals

if __name__ == '__main__':
	print "gc_binary.pyx"

	"""

	* Tutorial:

	To compute an isopleth for a binary system from simulations at different dmu2

	1. Instantiate an isopleth object from an array of histograms measured at different (mu1, dmu2)
	2. Call make_grid() to compute a grid over which to look for an isopleth.  This returns the grid so it can be manipulated by the user.
	3. Use get_iso() operate on those grids to find an isopleth desired

	* Notes:

	Histograms do not need to be provided in any order.

	* To Do:

	"""
