"""@docstring
@brief Manipulate binary systems sampled according to the order parameter N_tot
@author Nathan A. Mahynski
@date 02/07/2017
@filename gc_binary.pyx
"""

import gc_hist as gch
import numpy as np
import matplotlib.pyplot as plt
import copy, cython, types, operator, bisect, json
import scipy.ndimage

cimport cython
cimport numpy as np

from numpy import ndarray
from numpy cimport ndarray
from cpython cimport bool
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport fabs
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import UnivariateSpline

np.seterr(divide='raise', over='raise', invalid='raise', under='ignore')

@cython.boundscheck(False)
@cython.cdivision(True)
cdef _find_left_right(np.ndarray[np.double_t, ndim=1] ordered_dmu2, double val, bool bound=False):
	"""
	Find the indices to the left and right of value in an array ordered from low to high.
	When unbounded: if left == right, val is in the list, if < 0 or > len(ordered_dmu2) then is past the bounds.
	When bounded: returns values which are bounded by valid ordered_dmu2 array indices.

	Parameters
	----------
	ordered_dmu2 : ndarray
		Ordered array of dmu2 values from low to high
	val : double
		dmu2 to find bounds of
	bound : bool
		Whether or not to bound the left,right indices (default=False)

	Returns
	-------
	int, int
		left, right

	"""

	cdef int left, right
	cdef double tol = 1.0e-9

	if (val <= np.min(ordered_dmu2)):
		if (bound):
			left = 0
			right = 0
		else:
			left = -1
			right = -1
	elif (val >= np.max(ordered_dmu2)):
		if (bound):
			left = len(ordered_dmu2)-1
			right = len(ordered_dmu2)-1
		else:
			left = len(ordered_dmu2)
			right = len(ordered_dmu2)
	elif np.any([np.isclose(val, x) for x in ordered_dmu2]):
		x = np.where(np.abs(ordered_dmu2 - val) < tol)[0]
		if (len(x) != 1): raise Exception ('dmu2 values repeat, '+str(x)+' , '+str(ordered_dmu2)+' , '+str(val))
		left = x[0]
		right = left
	else:
		left = bisect.bisect(ordered_dmu2, val)-1
		right = left+1

	return left, right

@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _get_most_stable_phase (hist):
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
		free_energy[phase] = hist.data['thermo'][phase]['F.E./kT']
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
		dummy_sorted = sorted(dummy.items(), key=operator.itemgetter(0)) # Dictionary of {dmu2:histogram}

		self.data['dmu2'] = np.array([x[0] for x in dummy_sorted])
		self.data['histograms'] = [copy.deepcopy(x[1]) for x in dummy_sorted]

	def clear (self):
		"""
		Clears all information from the class.  Leaves metadata intact.

		"""

		self.data = {}

	def make_grid_multi (self, mu1_bounds, dmu2_bounds, delta, int p=1):
		"""
		Compute the discretized 2D (mu_1, dmu_2) isopleth surface in "chunks".

		Parameters
		----------
		mu1_bounds : array-like
			min, max of mu_1 to consider
		dmu2_bounds : array-like
			min, max of dmu_2 to consider
		delta : array-like
			Approximate width of mu bins to use in each (mu_1, dmu_2) dimension on a discrete grid. Usually ~ 0.2 is good balance between speed and accuracy.
		p : int
			Exponent to mix histograms with (default=1, "linear")

		Returns
		-------
		grid_x1 : ndarray
			2D array of x_1 (< 0 where thermodynamics could not be calculated)
		grid_mu : tuple
			Tuple of 2D arrays of (mu_1, dmu_2) at each "pixel"

		"""

		cdef int i, j, m, n, left, right

		if (not isinstance(mu1_bounds, (list, np.ndarray, tuple))): raise Exception ('Expects an array of mu1 bounds to construct isopleths')
		if (not isinstance(dmu2_bounds, (list, np.ndarray, tuple))): raise Exception ('Expects an array of dmu2 bounds to construct isopleths')
		if (not isinstance(delta, (list, np.ndarray, tuple))): raise Exception ('Expects an array of delta mu values to construct isopleths')

		if (len(mu1_bounds) != 2): raise Exception ('mu1_bound error in constructing isopleths')
		if (len(dmu2_bounds) != 2): raise Exception ('dmu2_bound error in constructing isopleths')
		if (len(delta) != 2): raise Exception ('delta error in constructing isopleths')

		if (mu1_bounds[1] <= mu1_bounds[0]): raise Exception ('mu1_bound error in constructing isopleths')
		if (dmu2_bounds[1] <= dmu2_bounds[0]): raise Exception ('dmu2_bound error in constructing isopleths')
		if (delta[0] <= 0): raise Exception ('delta error in constructing isopleths')
		if (delta[1] <= 0): raise Exception ('delta error in constructing isopleths')

		# Compute x1 at each point in the grid
		cdef int nx = np.ceil((mu1_bounds[1]-mu1_bounds[0])/delta[0])+1
		cdef int ny = np.ceil((dmu2_bounds[1]-dmu2_bounds[0])/delta[1])+1

		beta_targets = np.array([self.meta['beta']], dtype=np.float64)
		mu1_v = np.linspace(mu1_bounds[0],mu1_bounds[1],nx)
		dmu2_v = np.linspace(dmu2_bounds[0],dmu2_bounds[1],ny)
		self.data['X'], self.data['Y'] = np.meshgrid(mu1_v, dmu2_v)
		self.data['Z'] = np.zeros(self.data['X'].shape, dtype=np.float64)
		self.data['density'] = np.zeros(self.data['X'].shape, dtype=np.float64)
		self.data['F.E./kT'] = np.zeros(self.data['X'].shape, dtype=np.float64)

		# Compute which ones are left/right
		lr_matrix = np.zeros((len(dmu2_v), 2), dtype=np.int32)
		lr_weights = np.zeros((len(dmu2_v), 2), dtype=np.float64)
		for i in range(len(lr_matrix)):
			lr_matrix[i][0], lr_matrix[i][1] = _find_left_right(self.data['dmu2'], dmu2_v[i], True)

			# Mix these histograms
			dl = fabs(self.data['dmu2'][lr_matrix[i][0]] - dmu2_v[i])**p
			dr = fabs(self.data['dmu2'][lr_matrix[i][1]] - dmu2_v[i])**p
			if (dl + dr < 1.0e-9):
				# Have landed "exactly" on a dmu2 simulations so that left == right (or have gone past edge)
				assert (lr_matrix[i][0] == lr_matrix[i][1]), 'Unknown mixing distance error'
				lr_weights[i][0] = 1.0
				lr_weights[i][1] = 1.0
			else:
				lr_weights[i][0] = dr/(dr+dl) # Weights are "complementary" to distances
				lr_weights[i][1] = dl/(dr+dl) # Weights are "complementary" to distances

		# Consider all of mu1 space, one value at a time
		for i in range(len(mu1_v)):
			# Reweight all histograms first
			h_safe = np.array([True for m in range(len(self.data['histograms']))])
			for j in range(len(self.data['histograms'])):
				try:
					self.data['histograms'][j].reweight(mu1_v[i])
				except Exception as e:
					h_safe[j] = False

			# Extrapolate the histograms which are necessary (usually all of them) if they were safely reweighted
			h_matrix = np.array([[None for m in range(lr_matrix.shape[1])] for n in range(lr_matrix.shape[0])])
			for j in np.unique(lr_matrix):
				if (h_safe[j]):
					loc = np.where(lr_matrix == j)
					try:
						hists = self.data['histograms'][j].temp_dmu_extrap_multi(beta_targets, np.array([[x] for x in dmu2_v[loc[0]]]), self.meta['order'], self.meta['cutoff'], False, False)
					except Exception as e:
						print 'Error during extrapolation : '+str(e)
						pass
					else:
						h_matrix[loc] = hists[0]

			# Mix histograms that were successfully reweighted and extapolated, the compute x1
			for j in range(lr_matrix.shape[0]):
				if (not (h_matrix[j][0] is None or h_matrix[j][1] is None)):
					try:
						h_m = h_matrix[j][0].mix(h_matrix[j][1], lr_weights[j])
						h_m.thermo()
					except Exception as e:
						print 'Error during mixing and calculation : '+str(e)
						pass
					else:
						if (h_m.is_safe()):
							most_stable_phase = _get_most_stable_phase(h_m)
							self.data['Z'][j,i] = h_m.data['thermo'][most_stable_phase]['x1'] # j,i seem backward, but this is the way meshgrid works under default "xy" convention
							self.data['density'][j,i] = h_m.data['thermo'][most_stable_phase]['density']
							self.data['F.E./kT'][j,i] = h_m.data['thermo'][most_stable_phase]['F.E./kT']

		return self.data['Z'], (self.data['X'], self.data['Y'])

	def make_grid (self, mu1_bounds, dmu2_bounds, delta, float p=2.5):
		"""
		Compute the discretized 2D (mu_1, dmu_2) isopleth surface.

		Parameters
		----------
		mu1_bounds : array-like
			min, max of mu_1 to consider
		dmu2_bounds : array-like
			min, max of dmu_2 to consider
		delta : array-like
			Approximate width of mu bins to use in each (mu_1, dmu_2) dimension on a discrete grid
		p : float
			Exponent to mix histograms with (default=2.5)

		Returns
		-------
		grid_x1 : ndarray
			2D array of x_1 (< 0 where thermodynamics could not be calculated)
		grid_mu : tuple
			Tuple of 2D arrays of (mu_1, dmu_2) at each "pixel"

		"""

		cdef int i, j, left, right
		cdef double dmu2, mu1

		if (not isinstance(mu1_bounds, (list, np.ndarray, tuple))): raise Exception ('Expects an array of mu1 bounds to construct isopleths')
		if (not isinstance(dmu2_bounds, (list, np.ndarray, tuple))): raise Exception ('Expects an array of dmu2 bounds to construct isopleths')
		if (not isinstance(delta, (list, np.ndarray, tuple))): raise Exception ('Expects an array of delta mu values to construct isopleths')

		if (len(mu1_bounds) != 2): raise Exception ('mu1_bound error in constructing isopleths')
		if (len(dmu2_bounds) != 2): raise Exception ('dmu2_bound error in constructing isopleths')
		if (len(delta) != 2): raise Exception ('delta error in constructing isopleths')

		if (mu1_bounds[1] <= mu1_bounds[0]): raise Exception ('mu1_bound error in constructing isopleths')
		if (dmu2_bounds[1] <= dmu2_bounds[0]): raise Exception ('dmu2_bound error in constructing isopleths')
		if (delta[0] <= 0): raise Exception ('delta error in constructing isopleths')
		if (delta[1] <= 0): raise Exception ('delta error in constructing isopleths')

		# Compute x1 at each point in the grid
		cdef int nx = np.ceil((mu1_bounds[1]-mu1_bounds[0])/delta[0])+1
		cdef int ny = np.ceil((dmu2_bounds[1]-dmu2_bounds[0])/delta[1])+1

		cdef np.ndarray[np.double_t, ndim=1] mu1_v = np.linspace(mu1_bounds[0],mu1_bounds[1],nx)
		cdef np.ndarray[np.double_t, ndim=1] dmu2_v = np.linspace(dmu2_bounds[0],dmu2_bounds[1],ny)
		self.data['X'], self.data['Y'] = np.meshgrid(mu1_v, dmu2_v)
		self.data['Z'] = np.zeros(self.data['X'].shape, dtype=np.float64)
		self.data['density'] = np.zeros(self.data['X'].shape, dtype=np.float64)
		self.data['F.E./kT'] = np.zeros(self.data['X'].shape, dtype=np.float64)

		for i in range(self.data['X'].shape[0]):
			for j in range(self.data['X'].shape[1]):
				mu1 = self.data['X'][i,j]
				dmu2 = self.data['Y'][i,j]

				# Identify "bounding" dmu2's
				left, right = _find_left_right(self.data['dmu2'], dmu2, False)

				if (left == right):
					if (left < 0):
						# Below lower bound
						h_l = self.data['histograms'][0]
					elif (left == len(self.data['dmu2'])):
						# Above upper bound
						h_l = self.data['histograms'][-1]
					else:
						# Falls exactly on one of the dmu2 values
						h_l = self.data['histograms'][left]

					try:
						h_l.reweight(mu1)
						h_l = h_l.temp_dmu_extrap(self.meta['beta'], np.array([dmu2], dtype=np.float64), self.meta['order'], self.meta['cutoff'], False, True, False)
						h_l.thermo()
						if (not h_l.is_safe()):
							raise Exception ('extrapolated ln(PI) in histogram is not safe to use')
						else:
							# Find most stable phase and extract properties
							most_stable_phase = _get_most_stable_phase(h_l)
							self.data['Z'][i,j] = h_l.data['thermo'][most_stable_phase]['x1']
							self.data['density'][i,j] =  h_l.data['thermo'][most_stable_phase]['density']
							self.data['F.E./kT'][i,j] =  h_l.data['thermo'][most_stable_phase]['F.E./kT']
					except Exception as e:
						print 'Error at (mu_1,dmu_2) = ('+str(mu1)+','+str(dmu2)+') : '+str(e)+', continuing on...'
				else:
					# In between two measured dmu2 values
					h_l = self.data['histograms'][left]
					h_r = self.data['histograms'][right]

					try:
						h_l.reweight(mu1)
						h_l = h_l.temp_dmu_extrap(self.meta['beta'], np.array([dmu2], dtype=np.float64), self.meta['order'], self.meta['cutoff'], False, True, False)
						h_r.reweight(mu1)
						h_r = h_r.temp_dmu_extrap(self.meta['beta'], np.array([dmu2], dtype=np.float64), self.meta['order'], self.meta['cutoff'], False, True, False)
					except Exception as e:
						print 'Error at (mu_1,dmu_2) = ('+str(mu1)+','+str(dmu2)+') : '+str(e)+', continuing on...'
					else:
						# Mix these histograms
						dl = fabs(self.data['dmu2'][left] - dmu2)**p
						dr = fabs(self.data['dmu2'][right] - dmu2)**p
						wl = dr/(dr+dl) # Weights are "complementary" to distances
						wr = dl/(dr+dl) # Weights are "complementary" to distances

						h_m = h_l.mix(h_r, [wl, wr])
						try:
							h_m.thermo()
							if (not h_m.is_safe()):
								raise Exception ('extrapolated ln(PI) in histogram is not safe to use')
						except Exception as e:
							print 'Error at (mu_1,dmu_2) = ('+str(mu1)+','+str(dmu2)+') : '+str(e)+', continuing on...'
						else:
							# Find most stable phase and extract properties
							most_stable_phase = _get_most_stable_phase(h_m)
							self.data['Z'][i,j] = h_m.data['thermo'][most_stable_phase]['x1']
							self.data['density'][i,j] = h_m.data['thermo'][most_stable_phase]['density']
							self.data['F.E./kT'][i,j] = h_m.data['thermo'][most_stable_phase]['F.E./kT']

		return self.data['Z'], (self.data['X'], self.data['Y'])

	def dump (self, fname):
		"""
		Print surface to a json file to store.

		Parameters
		----------
		fname : str
			Filename to write data to

		"""

		f = open(fname, 'w')
		info = {}
		info['mu_1'] = self.data['X'].tolist()
		info['dmu_2'] = self.data['Y'].tolist()
		info['x_1'] = self.data['Z'].tolist()
		info['density'] = self.data['density'].tolist()
		info['F.E./kT'] = self.data['F.E./kT'].tolist()
		json.dump(info, f, sort_keys=True, indent=4)
		f.close()

	def zoom (self, factor, order=3, inplace=False):
		"""
		Resample the surface using cubic splines to smooth the isopleth. Does not change original values within the class.

		Parameters
		----------
		factor : float
			Zoom factor
		order : int
			Order of spline to use (default=3)
		inplace : bool
			If True, save the results internally and modify the class (default=False)

		Returns
		-------
		grid_x1 : ndarray
			2D array of x_1 at each "pixel"
		grid_mu : tuple
			Tuple of 2D arrays of (mu_1, dmu_2) at each "pixel"
		grid_rho : ndarray
			2D array of total (number) density at each "pixel"
		grid_fe : ndarray
			2D array of free energy/kT (of the most stable phase) at each "pixel"

		"""

		zx = scipy.ndimage.zoom(self.data['X'], factor, order=order)
		zy = scipy.ndimage.zoom(self.data['Y'], factor, order=order)
		zz = scipy.ndimage.zoom(self.data['Z'], factor, order=order)
		rho = scipy.ndimage.zoom(self.data['density'], factor, order=order)
		fe = scipy.ndimage.zoom(self.data['F.E./kT'], factor, order=order)

		if (inplace):
			self.data['X'] = zx
			self.data['Y'] = zy
			self.data['Z'] = zz
			self.data['density'] = rho
			self.data['F.E./kT'] = fe

		return zz, (zx, zy), rho, fe

def check_gibbs_duhem(np.ndarray[np.double_t, ndim=1] isobars, np.ndarray[np.double_t, ndim=2] grid_x1, np.ndarray[np.double_t, ndim=2] grid_p, np.ndarray[np.double_t, ndim=2] grid_mu1, np.ndarray[np.double_t, ndim=2] grid_dmu2):
	"""
	Compute the deviation from the Gibbs-Duhem equation.  For a binary system, only need to compute for 1 species.
	This assumes the provided grids have been obtained at some fixed temperature.

	Parameters
	----------
	isobars : ndarray
		Array of pressures to (isobars) to check consistency along
	grid_x1 : ndarray
		2D array of x_1 on (mu1, dmu2) grid
	grid_p : ndarray
		2D array of pressure on (mu1, dmu2) grid
	grid_mu1 : ndarray
		2D array of mu_1 at each "pixel"
	grid_dmu2 : ndarray
		2D array of dmu_2 at each "pixel"

	Returns
	-------
	error : list
		List of tuple of pressure, (x1_left, x1_right), and the relative error of the Gibbs-Duhem equation to the absolute value of the mean free energy per particle between each point along each isobar [(p_1, (x1_left,x1_right), |err|), (p_2, (x1_left,x1_right), |err|), ...].

	"""

	cdef double p, x1_bar, delta_x1, delta_mu1, delta_mu2, err
	cdef int i

	try:
		interp = RegularGridInterpolator((grid_dmu2[:,0], grid_mu1[0,:]), grid_x1, method='linear', bounds_error=False, fill_value=np.nan)
	except (Exception, TypeError, ValueError) as e:
		raise Exception ('Unable to create grid interpolator to check Gibbs-Duhem consistency : '+str(e))

	error = []
	for p in isobars:
		try:
			mu_vals_isobar = get_iso (p, grid_p, grid_mu1, grid_dmu2)
		except (Exception, TypeError, ValueError) as e:
			print ('Unable to check Gibbs-Duhem consistency along P = '+str(p)+' isobar : '+str(e))
			error.append((p, None))
		else:
			pts = np.array([(a[1], a[0]) for a in mu_vals_isobar])
			x1_vals = interp(pts)

			dmu1_dx1 = UnivariateSpline(x1_vals, [a[1] for a in mu_vals_isobar], k=3, s=0).derivative()
			dmu2_dx1 = UnivariateSpline(x1_vals, [a[0]+a[1] for a in mu_vals_isobar], k=3, s=0).derivative()

			error_p = []
			x1_t = []

			for x1v in x1_vals:
				if (not np.isnan(x1v)):
					err = x1v*dmu1_dx1(x1v) + (1.0-x1v)*dmu2_dx1(x1v)
					error_p.append(fabs(err))
					x1_t.append(x1v)

			error.append((p, x1_t, error_p))

			"""
			pts = np.array([(a[1], a[0]) for a in mu_vals_isobar])
			x1_vals = interp(pts)

			error_p = []
			x1_t = []

			for i in xrange(1, len(x1_vals)):
				if (not np.isnan(x1_vals[i]) and not np.isnan(x1_vals[i-1])):
					x1_bar = 0.5*(x1_vals[i] + x1_vals[i-1])
					delta_x1 = x1_vals[i] - x1_vals[i-1]
					delta_mu1 = pts[i][0] - pts[i-1][0]
					delta_mu2 = (pts[i][1] + pts[i][0]) - (pts[i-1][1] + pts[i-1][0])

					err = x1_bar*(delta_mu1/delta_x1) + (1.0-x1_bar)*(delta_mu2/delta_x1)
					error_p.append(fabs(err))
					x1_t.append((x1_vals[i],x1_vals[i-1]))

			error.append((p, x1_t, error_p))
			"""

	return error

def get_iso (double t, np.ndarray[np.double_t, ndim=2] grid_t, np.ndarray[np.double_t, ndim=2] grid_mu1, np.ndarray[np.double_t, ndim=2] grid_dmu2):
	"""
	Trace out the isopleth/isobar/iso- from the discretized grid of (mu_1, dmu_2).

	Parameters
	----------
	t : double
		Target mole fraction of species 1, pressure, etc.
	grid_t : ndarray
		2D array of x_1, P, etc. that you wish to find the curve along which this variable is constant
	grid_mu1 : ndarray
		2D array of mu_1 at each "pixel"
	grid_dmu2 : ndarray
		2D array of dmu_2 at each "pixel"

	Returns
	-------
	mu_vals : array
		array of tuples (mu_1, dmu_2) that trace out the isopleth

	"""

	cs = plt.contour(grid_mu1, grid_dmu2, grid_t, [t])
	p = cs.collections[0].get_paths()[0]
	v = p.vertices
	mu_vals = zip(v[:,0], v[:,1])

	return mu_vals

def parameterize_mesh (mu1_mesh, dmu2_mesh, x_mesh, y_mesh, x_pts):
	"""
	Parameterize one mesh in terms of another, e.g., rho (x) vs. P (y) on the (mu1, dmu2) grid.
	Can be used to parameterize thermodynamic properties along an isopleth.

	Parameters
	----------
	mu1_mesh : ndarray
		2D array of mu_1 at each "pixel" (from isopleth class)
	dmu2_mesh : ndarray
		2D array of dmu_1 at each "pixel" (from isopleth class)
	x_mesh : ndarray
		2D array of variable x defined on the mu1_mesh and dmu2_mesh
	y_mesh : ndarray
		2D array of variable y defined on the mu1_mesh and dmu2_mesh
	x_pts : array-like
		Array of tuples (mu_1, dmu_2) to estimate x and y (e.g. from get_iso() function)

	Returns
	-------
	xylist : list
		Array of parameterized (x,y) values

	"""

	if (mu1_mesh.shape != dmu2_mesh.shape): raise Exception ('Unequal grid sizes')
	if (x_mesh.shape != dmu2_mesh.shape): raise Exception ('Unequal grid sizes')
	if (x_mesh.shape != y_mesh.shape): raise Exception ('Unequal grid sizes')

	pts = np.array([(a[1], a[0]) for a in x_pts])
	x = mu1_mesh[0,:]
	y = dmu2_mesh[:,0]
	interp = RegularGridInterpolator((y, x), x_mesh, method='linear')
	x_vals = interp(pts)
	interp = RegularGridInterpolator((y, x), y_mesh, method='linear')
	y_vals = interp(pts)

	return zip(x_vals, y_vals)

def combine_isopleth_grids (mu1_arrays, dmu2_arrays, x1_arrays, rho_arrays=None, fe_arrays=None):
	"""
	Combine isopleth grids, assuming they are aligned along the dmu_2 axis.
	Assumes entries in mu1_arrays, dmu2_arrays are ordered from min to max, but the list of entries provided does not need to be.
	e.g., values of mu_1 ordered from min to max, and dmu_2 ordered from min to max.

	Parameters
	----------
	mu1_arrays : list
		Array of mu_1 grids to combine
	dmu2_arrays : list
		Array of dmu_2 grids to combine
	x1_arrays : list
		Array of x_1 grids to combine
	rho_arrays : list
		Array of density grids to combine (default=None)
	fe_arrays : list
		Array of free energy grids to combine (default=None)

	Returns
	-------
	grid_x1 : ndarray
		Composite 2D array of x_1
	grid_mu : tuple
		Tuple of composite 2D arrays of (mu_1, dmu_2) at each "pixel"
	grid_rho : ndarray
		Composite 2D array of density, if rho_arrays was given
	grid_fe : ndarray
		Composite 2D array of free energy, if fe_arrays was given

	"""

	if (not isinstance(mu1_arrays, (list, np.ndarray, tuple))): raise Exception ('Expects an array of mu1_arrays to combine isopleths')
	if (not isinstance(dmu2_arrays, (list, np.ndarray, tuple))): raise Exception ('Expects an array of dmu2_arrays to combine isopleths')
	if (not isinstance(x1_arrays, (list, np.ndarray, tuple))): raise Exception ('Expects an array of x1_arrays to combine isopleths')
	if (not (len(mu1_arrays) == len(dmu2_arrays) and len(dmu2_arrays) == len(x1_arrays))): raise Exception ('Must specify one mu_1, dmu_2, and x_1 for each isopleth')

	if (rho_arrays is not None):
		if (not isinstance(rho_arrays, (list, np.ndarray, tuple))): raise Exception ('Expects an array of rho_arrays to combine isopleths')
		if (not (len(mu1_arrays) == len(rho_arrays))): raise Exception ('Must specify one density for each isopleth')
	if (fe_arrays is not None):
		if (not isinstance(fe_arrays, (list, np.ndarray, tuple))): raise Exception ('Expects an array of fe_arrays to combine isopleths')
		if (not (len(mu1_arrays) == len(fe_arrays))): raise Exception ('Must specify one free energy for each isopleth')

	# Check that all individual isopleth grids have the same dimensions for mu1, dmu2, and x1
	for i in range(len(mu1_arrays)):
		if (not (mu1_arrays[i].shape == dmu2_arrays[i].shape and dmu2_arrays[i].shape == x1_arrays[i].shape)): raise Exception ('Each set of isopleth grids must have the same size')
		if (rho_arrays is not None):
			if (not (mu1_arrays[i].shape == rho_arrays[i].shape)): raise Exception ('Each set of isopleth grids must have the same size')
		if (fe_arrays is not None):
			if (not (mu1_arrays[i].shape == fe_arrays[i].shape)): raise Exception ('Each set of isopleth grids must have the same size')

	# Check that dmu2 dimensions across grids to combine are identical
	for i in range(len(mu1_arrays)-1):
		if (not (mu1_arrays[i].shape[0] == mu1_arrays[i+1].shape[0])): raise Exception ('dmu2 dimension not aligned')
		if (not (dmu2_arrays[i].shape[0] == dmu2_arrays[i+1].shape[0])): raise Exception ('dmu2 dimension not aligned')
		if (not (x1_arrays[i].shape[0] == x1_arrays[i+1].shape[0])): raise Exception ('dmu2 dimension not aligned')
		if (rho_arrays is not None):
			if (not (rho_arrays[i].shape[0] == rho_arrays[i+1].shape[0])): raise Exception ('dmu2 dimension not aligned')
		if (fe_arrays is not None):
			if (not (fe_arrays[i].shape[0] == fe_arrays[i+1].shape[0])): raise Exception ('dmu2 dimension not aligned')

	# Sort based on mu_1
	min_mu1 = [np.min(m1a) for m1a in mu1_arrays]
	if (fe_arrays is None and rho_arrays is None):
		zz = dict(list(enumerate(zip(min_mu1, mu1_arrays, dmu2_arrays, x1_arrays))))
	elif (fe_arrays is None and rho_arrays is not None):
		zz = dict(list(enumerate(zip(min_mu1, mu1_arrays, dmu2_arrays, x1_arrays, rho_arrays))))
	elif (fe_arrays is not None and rho_arrays is None):
		zz = dict(list(enumerate(zip(min_mu1, mu1_arrays, dmu2_arrays, x1_arrays, fe_arrays))))
	else:
		zz = dict(list(enumerate(zip(min_mu1, mu1_arrays, dmu2_arrays, x1_arrays, rho_arrays, fe_arrays))))
	sorted_zz = sorted(zz.items(), key=lambda x: x[1][0])

	X = copy.copy(sorted_zz[0][1][1])
	Y = copy.copy(sorted_zz[0][1][2])
	Z = copy.copy(sorted_zz[0][1][3])
	A = None
	B = None
	if (len(sorted_zz[0][1]) == 5):
		A = copy.copy(sorted_zz[0][1][4])
	elif (len(sorted_zz[0][1]) == 6):
		A = copy.copy(sorted_zz[0][1][4])
		B = copy.copy(sorted_zz[0][1][5])

	dmu2_ref = sorted_zz[0][1][2][:,1]
	for i in range(1, len(sorted_zz)):
		# Check dmu2 values are identical
		this_entry = sorted_zz[i]
		last_entry = sorted_zz[i-1]

		if (not np.all(np.abs(this_entry[1][2][:,0] - dmu2_ref) < 1.0e-9)): raise Exception ('dmu2 dimension not aligned')

		# Check mu1 limits don't overlap, else trim
		mu1_right =  this_entry[1][1][0,:]
		max_mu1_left = np.max(last_entry[1][1][0,:])
		ncols = bisect.bisect_left(mu1_right, max_mu1_left) # Which columns are duplicates?
		if (mu1_right[ncols] == max_mu1_left): ncols += 1 # Account for case where the two are equal at their edges

		# Concatenate
		X = np.concatenate((X,this_entry[1][1][:,ncols:]), axis=1) # Concatenate everything to the right of ncols
		Y = np.concatenate((Y,this_entry[1][2][:,ncols:]), axis=1) # Concatenate everything to the right of ncols
		Z = np.concatenate((Z,this_entry[1][3][:,ncols:]), axis=1) # Concatenate everything to the right of ncols
		if (len(sorted_zz[0][1]) == 5):
			A = np.concatenate((A,this_entry[1][4][:,ncols:]), axis=1)
		elif (len(sorted_zz[0][1]) == 6):
			A = np.concatenate((A,this_entry[1][4][:,ncols:]), axis=1)
			B = np.concatenate((B,this_entry[1][5][:,ncols:]), axis=1)

	if (A is None and B is None):
		return Z, (X, Y)
	elif (A is not None and B is None):
		return Z, (X, Y), A
	else:
		return Z, (X, Y), A, B

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
