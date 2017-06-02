"""@docstring
@brief Read and manipulate a 1D histogram from flat histogram simulations in grand canonical ensemble where N_tot is the order parameter
@author Nathan A. Mahynski
@date 07/18/2016
@filename gc_hist.pyx
"""

import operator, sys, copy, cython, types
import numpy as np

cimport cython
cimport numpy as np

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import fmin
from scipy.signal import argrelextrema
from netCDF4 import Dataset
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

@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double spec_exp(double a, double b):
	"""
	Compute the natural logarithm of the sum of a pair of exponentials.  i.e. ln(exp(a) + exp(b)).

	Parameters
	----------
	a : double
		Argument of first exponential
	b : double
		Argument of second exponential

	Returns
	-------
	double
		ln(exp(a) + exp(b))

	"""

	return double_max(a, b) + log(1.0 + exp(-fabs(a-b)))

@cython.boundscheck(False)
@cython.cdivision(True)
cdef _cython_normalize(self):
	"""
	Cythonized normalization of histogram. (histogram.normalize).

	"""

	cdef double lnNormPI = -sys.float_info.max
	cdef int i
	for i in xrange(len(self.data['ln(PI)'])):
		lnNormPI = spec_exp (lnNormPI, self.data['ln(PI)'][i])
	self.data['ln(PI)'] = self.data['ln(PI)'] - lnNormPI

@cython.boundscheck(False)
@cython.cdivision(True)
cdef _cython_reweight(self, double mu1_new):
	"""
	Cythonized normalization of histogram. (histogram.reweight).

	"""

	self.data['ln(PI)'] += (mu1_new - self.data['curr_mu'][0])*self.data['curr_beta']*self.data['ntot']
	self.normalize()

class histogram (object):
	"""
	Class which reads 1D histogram and computes the thermodynamic properties by reweighting, initialized from a netcdf4 file. This is designed to perform thermodynamic operations (reweighting, etc.) when N_tot is the order parameter from grand canonical ensemble.
	"""

	def __init__(self, fname, double beta_ref, mu_ref, int smooth=0, bool ke=False):
		"""
		Instatiate the class.

		Parameters
		----------
		fname : str
			netCDF4 file containing thermo information
		beta_ref : double
			1/kT These simulations were performed at
		mu_ref : array
			List/float of chemical potential(s) in order of species index this data was obtained at
		smooth : int
			Number of points to smooth over to find local extrema (default=0)
		ke : bool
			Flag which determines where kinetic energy was added to the U provided (default=False)

		"""

		self.metadata = {}
		self.metadata['beta_ref'] = beta_ref
		if (isinstance(mu_ref, list)):
			assert (len(mu_ref) > 0), 'Incomplete chemical potential information'
			self.metadata['mu_ref'] = np.array(mu_ref, dtype=np.float64)
		elif (isinstance(mu_ref, (float, np.float, np.float64, int, np.int32, np.int64))):
			self.metadata['mu_ref'] = np.array([mu_ref], dtype=np.float64)
		else:
			raise Exception ('Unrecognized type for mu_ref')
		self.metadata['nspec'] = len(self.metadata['mu_ref'])
		assert(self.metadata['beta_ref'] > 0), 'Illegal beta value'
		self.metadata['smooth'] = smooth
		assert(self.metadata['smooth'] >= 0), 'Illegal smooth value'
		self.metadata['smooth']
		assert(isinstance(fname, str)), 'Expects filename as a string'
		self.metadata['fname'] = fname
		self.metadata['used_ke'] = ke
		self.reload()

	def clear(self):
		"""
		Clear all data in the histogram, leaves metadata from init statement.

		"""

		self.data = {}

	def reload(self):
		"""
		(re)Load data from the netCDF4 file this object corresponds to.

		"""

		self.clear()
		self.data['curr_mu'] = copy.copy(self.metadata['mu_ref'])
		self.data['curr_beta'] = copy.copy(self.metadata['beta_ref'])
		self.data['nspec'] = copy.copy(self.metadata['nspec'])

		try:
			dataset = Dataset (self.metadata['fname'], "r", format="NETCDF4")
		except Exception as e:
			raise Exception ('Unable to load data from '+str(self.metadata['fname'])+' : '+str(e))

		self.metadata['file_history'] = copy.copy(dataset.history)
		self.data['ln(PI)'] = np.array(dataset.variables["ln(PI)"][:], dtype=np.float64)
		assert(dataset.nspec == self.metadata['nspec']), 'Different number of species in datafile from information initially specified'
		self.data['max_order'] = int(dataset.max_order)
		assert(self.data['max_order'] > 0), 'Error, max_order < 1'
		self.data['volume'] = float(dataset.volume)
		assert(self.data['volume'] > 0), 'Error, volume <= 0'
		self.data['ntot'] = np.array(dataset.variables["N_{tot}"][:], dtype=np.int)
		self.data['lb'] = self.data['ntot'][0]
		self.data['ub'] = self.data['ntot'][len(self.data['ntot'])-1]
		assert(self.data['lb'] < self.data['ub']), 'Error, bad bounds for N_tot'
		self.data['pk_hist'] = {}

		# check if pk hist data is available, but ok if not
		try:
			self.data['pk_hist']['hist'] = np.array(dataset.variables["P_{N_i}(N_{tot})"][:])
			self.data['pk_hist']['lb'] = np.array(dataset.variables["P_{N_i}(N_{tot})_{lb}"][:])
			self.data['pk_hist']['ub'] = np.array(dataset.variables["P_{N_i}(N_{tot})_{ub}"][:])
			self.data['pk_hist']['bw'] = np.array(dataset.variables["P_{N_i}(N_{tot})_{bw}"][:])
		except:
			pass

		# check if energy hist data is available, but ok if not
		self.data['e_hist'] = {}
		try:
			self.data['e_hist']['hist'] = np.array(dataset.variables["P_{U}(N_{tot})"][:])
			self.data['e_hist']['lb'] = np.array(dataset.variables["P_{U}(N_{tot})_{lb}"][:])
			self.data['e_hist']['ub'] = np.array(dataset.variables["P_{U}(N_{tot})_{ub}"][:])
			self.data['e_hist']['bw'] = np.array(dataset.variables["P_{U}(N_{tot})_{bw}"][:])
		except:
			pass

		self.data['mom'] = np.array(dataset.variables["N_{i}^{j}*N_{k}^{m}*U^{p}"][:])
		assert(self.data['mom'].shape == (self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1,len(self.data['ntot'])))

		dataset.close()

	def mix(self, other, weights):
		"""
		Create a new histogram that is a "mix" of this one and another one.
		Will only mix histograms at the same temperature, and chemical potentials.
		Properties, X, are mixed such that X(new) = [X(self)*weight_self + X(other)*weight_other]/[weight_self + weight_other].
		This requires that both histograms start from the same lower bound (i.e. Ntot = 0 usually), however
		it allows different upper bounds.  When these are different, the default is to only use the predictions
		from the histogram with the larger upper bound.
		All other settings are also taken from the histogram with the larger upper bound.

		Parameters
		----------
		other : gc_hist
			Other histogram to mix with this one
		weights : array
			Weights to assign to each of the histograms [weight_self, weight_other]

		Returns
		-------
		mixed : gc_hist
			Histogram that is a "blend" of these two

		"""

		cdef int max_idx, i, j, k, m, p
		cdef double tol = 1.0e-9

		# Ensure these histograms are compatible
		if (self.metadata['nspec'] != other.metadata['nspec']): raise Exception ('Difference in conditions, cannot mix histograms')
		if (self.metadata['used_ke'] != other.metadata['used_ke']): raise Exception ('Difference in conditions, cannot mix histograms')

		if (self.data['nspec'] != other.data['nspec']): raise Exception ('Difference in conditions, cannot mix histograms')
		if (fabs(self.data['curr_beta'] - other.data['curr_beta']) > tol): raise Exception ('Difference in conditions, cannot mix histograms')
		if (not np.all(np.abs(self.data['curr_mu'] - other.data['curr_mu']) < tol)): raise Exception ('Difference in conditions, cannot mix histograms')
		if (fabs(self.data['volume'] - other.data['volume']) > tol): raise Exception ('Difference in conditions, cannot mix histograms')
		if (self.data['max_order'] != other.data['max_order']): raise Exception ('Difference in conditions, cannot mix histograms')
		if (len(self.data['mom']) != len(other.data['mom'])): raise Exception ('Difference in conditions, cannot mix histograms')
		if (self.data['lb'] != other.data['lb']): raise Exception ('Difference in conditions, cannot mix histograms')

		if (not isinstance(weights, (np.ndarray, list, tuple))): raise Exception ('Requires 2 weights, cannot mix histograms')
		if (len(weights) != 2): raise Exception ('Requires 2 weights, cannot mix histograms')

		if (len(self.data['ln(PI)']) >= len(other.data['ln(PI)'])):
			longer_one = self
			max_idx = len(other.data['ln(PI)'])
		else:
			longer_one = other
			max_idx = len(self.data['ln(PI)'])

		mixed_hist = copy.deepcopy(longer_one)
		mixed_hist.data['file_history'] = 'this is a mixed histogram'

		# Since the raw data is provided at this condition, define the metadata manually for this mixed histogram.
		# This should allow this mixed hsitogram to be extrapolated/reweighted in the future, though that is not recommended.
		mixed_hist.metadata['fname'] = ''
		mixed_hist.metadata['beta_ref'] = mixed_hist.data['curr_beta']
		mixed_hist.metadata['mu_ref'] = mixed_hist.data['curr_mu']

		# Mix lnPI and moments
		mixed_hist.data['ln(PI)'] = mixed_hist.data['ln(PI)'].astype(np.float64) # guarantee not integer
		mixed_hist.data['ln(PI)'][:max_idx] = (self.data['ln(PI)'][:max_idx]*weights[0] + weights[1]*other.data['ln(PI)'][:max_idx])/(weights[0] + weights[1])

		mixed_hist.data['mom'] = mixed_hist.data['mom'].astype(np.float64) # guarantee not integer
		for i in range(self.data['nspec']):
			for j in range(self.data['max_order']+1):
				for k in range(self.data['nspec']):
					for m in range(self.data['max_order']+1):
						for p in range(self.data['max_order']+1):
							mixed_hist.data['mom'][i,j,k,m,p,:max_idx] = (self.data['mom'][i,j,k,m,p,:max_idx]*weights[0] + weights[1]*other.data['mom'][i,j,k,m,p,:max_idx])/(weights[0] + weights[1])

		# In the future, also mix e_hist and pk_hist, but for now delete them to prevent them from being used accidentally
		mixed_hist.data['pk_hist'] = {}
		mixed_hist.data['e_hist'] = {}

		return mixed_hist

	def normalize(self):
		"""
		Normalize the probability distribution.

		"""

		self._cy_normalize()

	def reweight(self, double mu1_target, print_screen=False):
		"""
		Reweight the lnPI distribution, etc. to different chemical potential of species 1.

		Normalizes the distribution after reweighting.

		Parameters
		----------
		mu1_target : double
			Desired chemical potentials of species 1
		print_screen : bool
			Print results to screen? (default=False)

		"""

		cdef int i
		cdef double dmu1 = mu1_target - self.data['curr_mu'][0]
		self._cy_reweight(mu1_target)
		self.data['curr_mu'] += dmu1
		if (print_screen):
			for i in xrange(len(self.data['ln(PI)'])):
				print i, self.data['ln(PI)'][i]-self.data['ln(PI)'][0]

	def _lowess_smooth(self, np.ndarray[np.double_t, ndim=1] x, np.ndarray[np.double_t, ndim=1] y, double frac):
		"""
		Smooth a vector using the lowess algorithm.

		Parameters
		----------
		x : ndarray
			x values, e.g., Ntot usually
		y : ndarray
			y values, e.g., ln(PI) usually
		frac : double
			fraction of the data to smooth over (0 < frac < 1)

		"""

		assert (frac > 0 and frac < 1), 'Bad fraction to smooth over'
		return lowess(y, x, frac=frac, it=0)

	def _butter_smooth(self):
		"""
		Smooth a vector using Butterworth filter.

		"""

		return

	def relextrema(self):
		"""
		Analyze the surface for locations of the local extrema.

		"""

		cdef int last_idx = len(self.data['ln(PI)'])-1, pos, i, l, r, lrmin, lrmax
		cdef double ave_q1, ave_q2

		if (last_idx <= 1):
			raise Exception ('ln(PI) not long enough to analyze for relative extrema')

		self.data['ln(PI)_maxima_idx'] = argrelextrema(self.data['ln(PI)'], np.greater, 0, self.metadata['smooth'], 'clip')[0]
		self.data['ln(PI)_minima_idx'] = argrelextrema(self.data['ln(PI)'], np.less, 0, self.metadata['smooth'], 'clip')[0]

		# argrelextrema does NOT include endpoints even if ln(PI) is a "straight" line, so manually include the bounds as well
		if (len(self.data['ln(PI)_maxima_idx']) > 0 and len(self.data['ln(PI)_minima_idx']) > 0):
			# Found some max and min
			if (0 not in self.data['ln(PI)_maxima_idx'] and 0 not in self.data['ln(PI)_minima_idx']):
				# Force "alternation" based on what occurs first (max or min?)
				if (self.data['ln(PI)_maxima_idx'][0] < self.data['ln(PI)_minima_idx'][0]):
					self.data['ln(PI)_minima_idx'] = np.append(0, self.data['ln(PI)_minima_idx'])
				elif (self.data['ln(PI)_maxima_idx'][0] > self.data['ln(PI)_minima_idx'][0]):
					self.data['ln(PI)_maxima_idx'] = np.append(0, self.data['ln(PI)_maxima_idx'])
				else:
					raise Exception ('Bad relative extrema calculation')

			if (last_idx not in self.data['ln(PI)_maxima_idx'] and last_idx not in self.data['ln(PI)_minima_idx']):
				# Force "alternation" based on what occured last (max or min?)
				if (self.data['ln(PI)_maxima_idx'][len(self.data['ln(PI)_maxima_idx'])-1] < self.data['ln(PI)_minima_idx'][len(self.data['ln(PI)_minima_idx'])-1]):
					self.data['ln(PI)_maxima_idx'] = np.append(self.data['ln(PI)_maxima_idx'], last_idx)
				elif (self.data['ln(PI)_maxima_idx'][len(self.data['ln(PI)_maxima_idx'])-1] > self.data['ln(PI)_minima_idx'][len(self.data['ln(PI)_minima_idx'])-1]):
					self.data['ln(PI)_minima_idx'] = np.append(self.data['ln(PI)_minima_idx'], last_idx)
				else:
					raise Exception ('Bad relative extrema calculation')
		elif (len(self.data['ln(PI)_maxima_idx']) > 0 and len(self.data['ln(PI)_minima_idx']) == 0):
			# Found at least one max, but no minima (e.g. supercritical with one peak in the middle and local minima at edges which are not detected by argrelextrema)
			# Therefore, assign minima indices to endpoints and between any local maxima missed because of "over smoothing"
			if (len(self.data['ln(PI)_maxima_idx']) > 1):
				added_minima = [0]
				for i in range(len(self.data['ln(PI)_maxima_idx'])-1):
					l = self.data['ln(PI)_maxima_idx'][i]
					r = self.data['ln(PI)_maxima_idx'][i+1]
					lmin = np.where(self.data['ln(PI)'][l:r] == np.min(self.data['ln(PI)'][l:r]))[0] + l
					added_minima.append(lmin)
				added_minima.append(len(self.data['ln(PI)'])-1)
				self.data['ln(PI)_minima_idx'] = np.array(added_minima)
			else:
				# Only one local maxima in the interior of ln(PI)
				self.data['ln(PI)_minima_idx'] = np.array([0, len(self.data['ln(PI)'])-1])
		elif (len(self.data['ln(PI)_maxima_idx']) == 0 and len(self.data['ln(PI)_minima_idx']) > 0):
			# Found at least one minima, but no maxima.
			# Therefore, assign maxima indices to endpoints and between any local minima missed because of "over smoothing"
			if (len(self.data['ln(PI)_minima_idx']) > 1):
				added_maxima = [0]
				for i in range(len(self.data['ln(PI)_minima_idx'])-1):
					l = self.data['ln(PI)_minima_idx'][i]
					r = self.data['ln(PI)_minima_idx'][i+1]
					lmax = np.where(self.data['ln(PI)'][l:r] == np.max(self.data['ln(PI)'][l:r]))[0] + l
					added_maxima.append(lmax)
				added_maxima.append(len(self.data['ln(PI)'])-1)
				self.data['ln(PI)_maxima_idx'] = np.array(added_maxima)
			else:
				# Only one local minima in the interior of ln(PI)
				self.data['ln(PI)_maxima_idx'] = np.array([0, len(self.data['ln(PI)'])-1])
		else:
			# Skewed strongly one way or the other ("straight line")
			# self.is_safe() will throw error if positive "slope", but for now assume the user actually wants this calculation or has negative slope, i.e. mu --> -inf
			self.data['ln(PI)_maxima_idx'] = np.where(self.data['ln(PI)'] == np.max(self.data['ln(PI)']))[0]
			self.data['ln(PI)_minima_idx'] = np.where(self.data['ln(PI)'] == np.min(self.data['ln(PI)']))[0]

		"""
		# For now, just compare neighbors
		if (0 not in self.data['ln(PI)_maxima_idx'] and 0 not in self.data['ln(PI)_minima_idx']):
			if (self.data['ln(PI)'][0] < self.data['ln(PI)'][1]):
				self.data['ln(PI)_minima_idx'] = np.append(0, self.data['ln(PI)_minima_idx'])
			else:
				self.data['ln(PI)_maxima_idx'] = np.append(0, self.data['ln(PI)_maxima_idx'])
		if (last_idx not in self.data['ln(PI)_maxima_idx'] and last_idx not in self.data['ln(PI)_minima_idx']):
			if (self.data['ln(PI)'][last_idx] < self.data['ln(PI)'][last_idx-1]):
				self.data['ln(PI)_minima_idx'] = np.append(self.data['ln(PI)_minima_idx'], last_idx)
			else:
				self.data['ln(PI)_maxima_idx'] = np.append(self.data['ln(PI)_maxima_idx'], last_idx)
		"""

		# Check that maxima and minima alternate
		if (not (np.abs(len(self.data['ln(PI)_maxima_idx']) - len(self.data['ln(PI)_minima_idx'])) <= 1)):
			raise Exception('There are '+str(len(self.data['ln(PI)_maxima_idx']))+' local maxima and '+str(len(self.data['ln(PI)_minima_idx']))+' local minima, so cannot be alternating, try adjusting the value of smooth')

		order = np.zeros(len(self.data['ln(PI)_maxima_idx']) + len(self.data['ln(PI)_minima_idx']))
		if (self.data['ln(PI)_maxima_idx'][0] < self.data['ln(PI)_minima_idx'][0]):
			order[::2] = self.data['ln(PI)_maxima_idx']
			order[1::2] = self.data['ln(PI)_minima_idx']
		else:
			order[::2] = self.data['ln(PI)_minima_idx']
			order[1::2] = self.data['ln(PI)_maxima_idx']

		if (not (np.all([order[i] <= order[i+1] for i in xrange(len(order)-1)]))):
			raise Exception ('Local maxima and minima not sorted correctly, try adjusting the value of smooth (max,min) = '+str(self.data['ln(PI)_maxima_idx'])+', '+str(self.data['ln(PI)_minima_idx'])+' : '+str(order.tolist()))

	def coexisting (self, rtol=1.0e-3):
		"""
		Search for all phases that are in equilibrium (equal free energy) and return the indices in self.data['thermo'] that correspond to them.
		If either only one phase exists or if no phases in equilibrium, returns empty matrix.

		Parameters
		----------
		rtol : double
			Relative tolerance of F.E./kT to define being equal (default=1.0e-3 or 0.1%)

		Returns
		-------
		array
			(1,p) array of coexisting phases, indices of phases at equilibrium

		"""

		if ('thermo' not in self.data):
			raise Exception ('Thermodynamic properties should be called first (self.thermo())')

		if (len(self.data['thermo']) == 1):
			return [[]]
		else:
			eq = []
			for i in range(len(self.data['thermo'])):
				x = [i]
				for j in range(i+1, len(self.data['thermo'])):
					if (fabs((self.data['thermo'][i]['F.E./kT'] - self.data['thermo'][j]['F.E./kT'])/self.data['thermo'][i]['F.E./kT']) < rtol):
						x.append(j)
				if (len(x) > 1):
					eq.append(x)

			return eq

	def thermo(self, bool props=True, bool complete=False, collect=None):
		"""
		Integrate the lnPI distribution, etc. and compute average thermodynamic properties of each phase identified.

		This adds F.E./kT, nn_mom, un_mom, n1, n2, ..., x1, x2, ..., u, and density keys to data['thermo'][phase_idx] for each phase. Does not check "safety" of the calculation, use is_safe() for that.
		Note that pressure, P = -1*[F.E./kT]/V/beta.

		Parameters
		----------
		props : bool
			If True then computes the extensive properties, otherwise just integrates lnPI (free energy) for each phase (default=True)
		complete : bool
			If True then compute properties of entire distribution, ignoring phase segmentation of lnPI surface (default=False)
		collect : function (histogram)
			Function that will "collect" maxima and minima into phases if each peak does not represent a phase, which takes a keyword argument "hist" to accept this class.
			This function must modify the class and report a maxima, boudned by minima, for each resulting phase. See example function in collect.py called janus_collect(). (default=None)

		"""

		cdef int p, i, j, k, m, q, left, right, min_ctr = 0, nphases = 0
		cdef double lnX, sum_prob

		if (not complete):
			try:
				self.normalize()
			except Exception as e:
				raise Exception ('Unable to normalize ln(PI) : '+str(e))

			try:
				self.relextrema()
			except Exception as e:
				raise Exception ('Unable to find relative extrema : '+str(e))

			# collect extrema if necessary
			if (collect is not None):
				collect(hist=self)

			nphases = len(self.data['ln(PI)_maxima_idx'])
		else:
			try:
				self.normalize()
			except Exception as e:
				raise Exception ('Unable to normalize ln(PI) : '+str(e))
			nphases = 1

		phase = {}

		for p in xrange(nphases):
			phase[p] = {}

			if (not complete):
				if (self.data['ln(PI)_maxima_idx'][p] > 0):
					# max occurs to the right of the lower bound
					left = self.data['ln(PI)_minima_idx'][min_ctr] # on first iteration this should be lower bound
					min_ctr += 1
				else:
					# max occurs at left bound, so only integrate to the right
					left = 0

				if (self.data['ln(PI)_maxima_idx'][p] < len(self.data['ln(PI)'])-1):
					right = self.data['ln(PI)_minima_idx'][min_ctr]
				else:
					right = len(self.data['ln(PI)'])

				# formally include final endpoint
				if (right == len(self.data['ln(PI)'])-1):
					right += 1
			else:
				left = 0
				right = len(self.data['ln(PI)'])

			# report with respect to reference state of lowest bound (no longer normalized)
			lnX = -sys.float_info.max
			for j in xrange(left, right):
				lnX = spec_exp (lnX, self.data['ln(PI)'][j]-self.data['ln(PI)'][0])
			phase[p]['F.E./kT'] = -lnX
			phase[p]['bound_idx'] = (left,right)

			if (props):
				prob = np.exp(self.data['ln(PI)'][left:right])
				sum_prob = np.sum(prob)

				# average the moments vector to get properties of this phase
				phase[p]['mom'] = np.zeros((self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1), dtype=np.float64)
				for i in xrange(self.data['nspec']):
					for j in xrange(self.data['max_order']+1):
						for k in xrange(self.data['nspec']):
							for m in xrange(self.data['max_order']+1):
								for q in xrange(self.data['max_order']+1):
									x = self.data['mom'][i,j,k,m,q,left:right]
									phase[p]['mom'][i,j,k,m,q] = np.sum(prob*x)/sum_prob

				# store <Ntot>,<N1>,<N2>,...,<U>,<x1>,<x2>,...
				nsum = 0.0
				for i in xrange(self.data['nspec']):
					phase[p]['n'+str(i+1)] = phase[p]['mom'][i,1,0,0,0]
					nsum += phase[p]['mom'][i,1,0,0,0]
				phase[p]['ntot'] = nsum
				phase[p]['density'] = nsum/self.data['volume']
				phase[p]['u'] = phase[p]['mom'][0,0,0,0,1]
				for i in xrange(self.data['nspec']):
					phase[p]['x'+str(i+1)] = phase[p]['mom'][i,1,0,0,0]/nsum

		self.data['thermo'] = phase

	def is_safe(self, double cutoff=10.0, bool complete=False):
		"""
		Check that the current ln(PI) distribution extends far enough to trust a thermo() calculation.
		First do thermo() calculation, then check is_safe().
		This can be thrown off by poorly smoothed data, so may have to manually check.

		Parameters
		----------
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10.0)
		complete : bool
			If True then consider the entire distribution, ignoring phase segmentation of lnPI surface (default=False)

		"""

		if (not complete):
			try:
				nphases = len(self.data['ln(PI)_maxima_idx'])
			except KeyError:
				try:
					self.normalize()
				except Exception as e:
					raise Exception ('Unable to normalize ln(PI) : '+str(e))

				try:
					self.relextrema()
				except Exception as e:
					raise Exception ('Unable to find relative extrema in ln(PI) : '+str(e))

			# since phases are ordered in increasing Ntot, just compare last maxima to end of ln(PI) distribution
			maxima = self.data['ln(PI)'][self.data['ln(PI)_maxima_idx']]

			if (maxima[len(maxima)-1] - self.data['ln(PI)'][len(self.data['ln(PI)'])-1] < cutoff):
				return False
			else:
				return True
		else:
			if (np.max(self.data['ln(PI)']) - self.data['ln(PI)'][len(self.data['ln(PI)'])-1] < cutoff):
				return False
			else:
				return True

	def find_phase_eq(self, double lnZ_tol, double mu_guess, double beta=0.0, object dMu=[], int extrap_order=1, double cutoff=10.0, bool override=False, bool reterr=False, bool first_order_mom=False, collect=None):
		"""
		Search for coexistence between two phases which have a "width" of at least the size of self.metadata['smooth'].

		Creates a local copy of self so self is NOT modified by this search.

		Parameters
		----------
		lnZ_tol : double
			Permissible difference in integral of lnZ (free energy/kT) between phases in equilibrium.
		mu_guess : double
			Chemical potential of species 1 to start iterating from.
		beta : double
			Temperature at which to search for equilibrium.  If <=0 uses curr_beta so no extrapolation. (default=0.0)
		dMu : ndarray
			Target values of [mu_2-mu_1, mu_3-mu_1, ..., mu_N-mu_1].  If dMU == [], assumes the current dMu's are to be used. (default=[])
		extrap_order : int
			Order of extrapolation to use if going to a different temperature. (default=1)
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting. (default=10.0)
		override : bool
			Override warnings about inaccuracies in lnPI after temperature extrapolation at coexistence. (default=False)
		reterr : bool
			Return the error associated with numerical optimization. (default=False)
		first_order_mom : bool
			If True, only use first order extrapolation to extrapolate moments when using higher order extrapolation of ln(PI) (default=False)
		collect : function (histogram)
			Function that will "collect" maxima and minima into phases if each peak does not represent a phase, which takes a keyword argument "hist" to accept this class.
			This function must modify the class and report a maxima, boudned by minima, for each resulting phase. See example function in collect.py called janus_collect(). (default=None)

		Returns
		-------
		histogram, err (optional)
			Copy of this histogram, but reweighted to the point of phase coexistence, error if requested

		"""
		# Clone self to avoid any changes

		tmp_hist = copy.deepcopy(self)
		curr_dMu = np.array([self.data['curr_mu'][i] - self.data['curr_mu'][0] for i in xrange(1,self.data['nspec'])], dtype=np.float64)

		if (len(dMu) == 0):
			# Assume no change to the dMu values
			new_dMu = copy.copy(curr_dMu)
		else:
			# Must be specified for all components
			assert (len(dMu) == self.data['nspec']-1), 'Need to specify dMu for components 2-N'
			new_dMu = copy.copy(np.array(dMu, dtype=np.float64))

		if (beta <= 0.0):
			beta = self.data['curr_beta']

		# Search for equilibrium
		tmp_hist.normalize()
		full_out = fmin(phase_eq_error, mu_guess, ftol=lnZ_tol, args=(tmp_hist,beta,new_dMu,extrap_order,cutoff,True,tmp_hist.metadata['smooth'],collect=collect), maxfun=100000, maxiter=100000, full_output=True, disp=True, retall=True)
		if (full_out[:][4] != 0): # full_out[:][4] = warning flag
			raise Exception ('Error, unable to locate phase coexistence : '+str(full_out))

		try:
			tmp_hist.reweight(full_out[0][0])
			if (beta != self.data['curr_beta'] or np.all(new_dMu == curr_dMu) == False):
				tmp_hist.temp_dmu_extrap(beta, new_dMu, extrap_order, cutoff, override, False, False, first_order_mom)
			tmp_hist.thermo(collect=collect)
		except Exception as e:
			raise Exception ('Found coexistence, but unable to compute properties afterwards: '+str(e))

		if (reterr):
			return tmp_hist, full_out[1]
		else:
			return tmp_hist

	def temp_extrap(self, double target_beta, int order=1, double cutoff=10.0, override=False, clone=True, skip_mom=False):
		"""
		Use temperature extrapolation to estimate lnPI and other extensive properties from current conditions.

		Should do reweighting (if desired) first, then call this member (curr_mu1 should reflect this).

		Parameters
		----------
		target_beta : double
			1/kT of the desired distribution
		order : int
			Order of the extapolation to use (default=1)
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10)
		override : bool
			Override warnings about inaccuracies in lnPI (default=False)
		clone : bool
			If True, creates a copy of self and extrapolates the copy so this object is not modified, else extrapolates self (default=True)
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		Returns
		-------
		histogram
			Cloned object with information extrapolated to new 1/kT value, histogram is renormalized

		"""

		if (np.abs(self.metadata['beta_ref'] - self.data['curr_beta']) > 1.0e-6):
			raise Exception ('Cannot extrapolate the same histogram class twice')

		if (not skip_mom):
			needed_order = order+1
		else:
			needed_order = order

		if (self.data['max_order'] < needed_order):
			raise Exception ('Maximum order stored in simulation not high enough to calculate this order of extrapolation')

		if (clone):
			tmp_hist = copy.deepcopy(self)
		else:
			tmp_hist = self

		# For numerical stability, always normalize lnPI before extrapolating
		tmp_hist.normalize()

		if (order == 1):
			try:
				tmp_hist._temp_extrap_1(target_beta, cutoff, override, skip_mom)
			except Exception as e:
				raise Exception('Unable to extrapolate in temperature: '+str(e))
		elif (order == 2):
			try:
				tmp_hist._temp_extrap_2(target_beta, cutoff, override, skip_mom)
			except Exception as e:
				raise Exception('Unable to extrapolate in temperature: '+str(e))
		elif (order == 3):
			try:
				tmp_hist._temp_extrap_3(target_beta, cutoff, override, skip_mom)
			except Exception as e:
				raise Exception('Unable to extrapolate in temperature: '+str(e))
		else:
			raise Exception('No implementation for temperature extrapolation of order '+str(order))

		tmp_hist.data['curr_beta'] = target_beta

		# Renormalize afterwards as well
		tmp_hist.normalize()

		return tmp_hist

	def dmu_extrap(self, np.ndarray[np.double_t, ndim=1] target_dmu, int order=1, double cutoff=10.0, override=False, clone=True, skip_mom=False):
		"""
		Use delta Mu extrapolation to estimate lnPI and other extensive properties from current conditions.

		Should do reweighting (if desired) first, then call this member (curr_mu1 should reflect this).

		Parameters
		----------
		target_dmu : ndarray
			Desired difference of chemical potentials of species 2-N, dmu_i = mu_i - mu_1
		order : int
			Order of the extapolation to use (default=1)
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10)
		override : bool
			Override warnings about inaccuracies in lnPI (default=False)
		clone : bool
			If True, creates a copy of self and extrapolates the copy so this object is not modified, else extrapolates self (default=True)
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		Returns
		-------
		histogram
			Cloned object with information extrapolated to new 1/kT value, histogram is renormalized

		"""

		assert (len(target_dmu) == self.data['nspec']-1), 'Must specify delta mu for all components 2-N'

		orig_dmu = self.metadata['mu_ref'][1:] - self.metadata['mu_ref'][0]
		curr_dmu = self.data['curr_mu'][1:] - self.data['curr_mu'][0]
		if (np.any(np.abs(orig_dmu - curr_dmu) > 1.0e-6)):
			raise Exception ('Cannot extrapolate the same histogram class twice')

		if (not skip_mom):
			needed_order = order+1
		else:
			needed_order = order

		if (self.data['max_order'] < needed_order):
			raise Exception ('Maximum order stored in simulation not high enough to calculate this order of extrapolation')

		if (clone):
			tmp_hist = copy.deepcopy(self)
		else:
			tmp_hist = self

		# For numerical stability, always normalize lnPI before extrapolating
		tmp_hist.normalize()

		if (order == 1):
			try:
				tmp_hist._dmu_extrap_1(target_dmu, cutoff, override, skip_mom)
			except Exception as e:
				raise Exception('Unable to extrapolate in dMu: '+str(e))
		elif (order == 2):
			try:
				tmp_hist._dmu_extrap_2(target_dmu, cutoff, override, skip_mom)
			except Exception as e:
				raise Exception('Unable to extrapolate in dMu: '+str(e))
		else:
			raise Exception('No implementation for dMu extrapolation of order '+str(order))

		tmp_hist.data['curr_mu'][1:] = tmp_hist.data['curr_mu'][0] + target_dmu

		# Renormalize afterwards as well
		tmp_hist.normalize()

		return tmp_hist

	def temp_dmu_extrap_multi(self, np.ndarray[np.double_t, ndim=1] target_betas, np.ndarray[np.double_t, ndim=2] target_dmus, int order=1, double cutoff=10.0, override=False, skip_mom=False, bool first_order_mom=False):
		"""
		Use temperature and delta Mu extrapolation to estimate lnPI and other extensive properties from current conditions for a grid different beta and dMu values.
		Creates a 2D grid of conditions where beta is the first dimension, and dMu's are the second.  So all target_betas and target_dmus are evaluated.
		By default, this first makes a clone of itself so this extrapolation does not modify the original histogram object.

		Should do reweighting (if desired) first, then call this member (curr_mu1 should reflect this).

		Parameters
		----------
		target_betas : ndarray
			Array of 1/kT of the desired distribution
		target_dmus : ndarray
			Array of  [desired difference of chemical potentials of species 2-N, dmu_i = mu_i - mu_1]
		order : int
			Order of the extapolation to use (default=1)
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10)
		override : bool
			Override warnings about inaccuracies in lnPI (default=False)
		skip_mom : bool
			Skip extrapolation of moments (default=False)
		first_order_mom : bool
			If True, only use first order extrapolation to extrapolate moments when using higher order extrapolation of ln(PI) (default=False)

		Returns
		-------
		array
			2D array of histogram objects with information extrapolated to [1/kT, dMu] values. Histograms are renormalized.  If an extrapolation failed, the array is None at that position.

		"""

		cdef int i, j

		if (np.abs(self.metadata['beta_ref'] - self.data['curr_beta']) > 1.0e-6):
			raise Exception ('Cannot extrapolate the same histogram class twice')

		for target_dmu in target_dmus:
			assert (len(target_dmu) == self.data['nspec']-1), 'Must specify delta mu for all components 2-N'

		orig_dmu = copy.copy(self.metadata['mu_ref'][1:] - self.metadata['mu_ref'][0])
		curr_dmu = copy.copy(self.data['curr_mu'][1:] - self.data['curr_mu'][0])
		if (np.any(np.abs(orig_dmu - curr_dmu) > 1.0e-6)):
			raise Exception ('Cannot extrapolate the same histogram class twice')

		if (not skip_mom):
			needed_order = order+1
		else:
			needed_order = order

		if (self.data['max_order'] < needed_order):
			raise Exception ('Maximum order stored in simulation not high enough to calculate this order of extrapolation')

		if (order == 1):
			try:
				hists = self._temp_dmu_extrap_1_multi(target_betas, target_dmus, cutoff, override, skip_mom)
			except Exception as e:
				raise Exception('Unable to extrapolate : '+str(e))
		elif (order == 2):
			try:
				hists = self._temp_dmu_extrap_2_multi(target_betas, target_dmus, cutoff, override, skip_mom, first_order_mom)
			except Exception as e:
				raise Exception('Unable to extrapolate : '+str(e))
		else:
			raise Exception('No implementation for temperature + dMu extrapolation of order '+str(order))

		for i in range(len(target_betas)):
			for j in range(len(target_dmus)):
				hists[i][j].data['curr_beta'] = copy.copy(target_betas[i])
				hists[i][j].data['curr_mu'][1:] = copy.copy(hists[i][j].data['curr_mu'][0] + target_dmus[j])

				# Renormalize afterwards as well
				hists[i][j].normalize()

		return hists

	def temp_dmu_extrap(self, double target_beta, np.ndarray[np.double_t, ndim=1] target_dmu, int order=1, double cutoff=10.0, override=False, bool clone=True, bool skip_mom=False, bool first_order_mom=False):
		"""
		Use temperature and delta Mu extrapolation to estimate lnPI and other extensive properties from current conditions.

		Should do reweighting (if desired) first, then call this member (curr_mu1 should reflect this).

		Parameters
		----------
		target_beta : double
			1/kT of the desired distribution
		target_dmu : ndarray
			Desired difference of chemical potentials of species 2-N, dmu_i = mu_i - mu_1
		order : int
			Order of the extapolation to use (default=1)
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10)
		override : bool
			Override warnings about inaccuracies in lnPI (default=False)
		clone : bool
			If True, creates a copy of self and extrapolates the copy so this object is not modified, else extrapolates self (default=True)
		skip_mom : bool
			Skip extrapolation of moments (default=False)
		first_order_mom : bool
			If True, only use first order extrapolation to extrapolate moments when using higher order extrapolation of ln(PI) (default=False)

		Returns
		-------
		histogram
			Cloned object with information extrapolated to new 1/kT value, histogram is renormalized

		"""

		if (np.abs(self.metadata['beta_ref'] - self.data['curr_beta']) > 1.0e-6):
			raise Exception ('Cannot extrapolate the same histogram class twice')

		assert (len(target_dmu) == self.data['nspec']-1), 'Must specify delta mu for all components 2-N'

		orig_dmu = copy.copy(self.metadata['mu_ref'][1:] - self.metadata['mu_ref'][0])
		curr_dmu = copy.copy(self.data['curr_mu'][1:] - self.data['curr_mu'][0])
		if (np.any(np.abs(orig_dmu - curr_dmu) > 1.0e-6)):
			raise Exception ('Cannot extrapolate the same histogram class twice')

		if (not skip_mom):
			needed_order = order+1
		else:
			needed_order = order

		if (self.data['max_order'] < needed_order):
			raise Exception ('Maximum order stored in simulation not high enough to calculate this order of extrapolation')

		if (clone):
			tmp_hist = copy.deepcopy(self)
		else:
			tmp_hist = self

		# For numerical stability, always normalize lnPI before extrapolating
		tmp_hist.normalize()

		if (order == 1):
			try:
				tmp_hist._temp_dmu_extrap_1(target_beta, target_dmu, cutoff, override, skip_mom)
			except Exception as e:
				raise Exception('Unable to extrapolate : '+str(e))
		elif (order == 2):
			try:
				tmp_hist._temp_dmu_extrap_2(target_beta, target_dmu, cutoff, override, skip_mom, first_order_mom)
			except Exception as e:
				raise Exception('Unable to extrapolate : '+str(e))
		else:
			raise Exception('No implementation for temperature + dMu extrapolation of order '+str(order))

		tmp_hist.data['curr_beta'] = target_beta
		tmp_hist.data['curr_mu'][1:] = copy.copy(tmp_hist.data['curr_mu'][0] + target_dmu)

		# Renormalize afterwards as well
		tmp_hist.normalize()

		return tmp_hist

	def _temp_dmu_extrap_1_multi(self, np.ndarray[np.double_t, ndim=1] target_betas, np.ndarray[np.double_t, ndim=2] target_dmus, double cutoff=10.0, override=False, skip_mom=False):
		"""
		Extrapolate the histogam in an array of different temperatures and dMus using first order corrections.

		Parameters
		----------
		target_betas : ndarray
			Array of 1/kT of the desired distribution
		target_dmus : ndarray
			Array of [desired difference of chemical potentials of species 2-N, dmu_i = mu_i - mu_1]
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10)
		override : bool
			Override warnings about inaccuracies in lnPI (default=False)
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		Returns
		-------
		array
			2D array of histogram clones at each [beta, dMu]

		"""

		cdef double dB
		cdef np.ndarray[np.double_t, ndim=1] target_dDmu
		cdef np.ndarray[np.double_t, ndim=2] dlnPI = np.zeros((self.data['nspec']-1, self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=7] dm = np.zeros((self.data['nspec']-1, self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef int i, j, k, m, p, q

		hists = []

		# Extrapolate lnPI
		if (not override):
			# If histogram does not extend far enough, cannot calculate average quantities needed for extrapolation accurately
			assert (np.max(self.data['ln(PI)']) - cutoff > self.data['ln(PI)'][len(self.data['ln(PI)'])-1]), 'Error, histogram edge effect encountered in temperature extrapolation'

		try:
			# For numerical stability, always normalize lnPI before extrapolating
			cc = copy.deepcopy(self)
			cc.normalize()
			dlnPI, dm = cc._dBMU(skip_mom)
		except:
			raise Exception ('Unable to compute first derivative')

		for target_beta in target_betas:
			dB = target_beta - self.data['curr_beta']
			hc = []

			for target_dmu in target_dmus:
				target_dDmu = target_dmu - (self.data['curr_mu'][1:] - self.data['curr_mu'][0])

				try:
					clone = copy.deepcopy(self)

					clone.data['ln(PI)'] += dB*dlnPI[0]
					for q in xrange(1,clone.data['nspec']):
						clone.data['ln(PI)'] += target_dDmu[q-1]*dlnPI[q]

					for i in xrange(clone.data['nspec']):
						for j in xrange(clone.data['max_order']+1):
							for k in xrange(clone.data['nspec']):
								for m in xrange(clone.data['max_order']+1):
									for p in xrange(clone.data['max_order']+1):
										clone.data['mom'][i,j,k,m,p] += dB*dm[0,i,j,k,m,p]
										for q in xrange(1, clone.data['nspec']):
											clone.data['mom'][i,j,k,m,p] += target_dDmu[q-1]*dm[q,i,j,k,m,p]

				except:
					hc.append(None)
				else:
					hc.append(clone)

			hists.append(hc)

		return hists

	def _temp_dmu_extrap_1(self, double target_beta, np.ndarray[np.double_t, ndim=1] target_dmu, double cutoff=10.0, override=False, skip_mom=False):
		"""
		Extrapolate the histogam in temperature and dMu using first order corrections.

		Parameters
		----------
		target_beta : double
			1/kT of the desired distribution
		target_dmu : ndarray
			Desired difference of chemical potentials of species 2-N, dmu_i = mu_i - mu_1
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10)
		override : bool
			Override warnings about inaccuracies in lnPI (default=False)
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		"""

		cdef double dB = target_beta - self.data['curr_beta']
		cdef np.ndarray[np.double_t, ndim=1] target_dDmu = target_dmu - (self.data['curr_mu'][1:] - self.data['curr_mu'][0])
		cdef np.ndarray[np.double_t, ndim=2] dlnPI = np.zeros((self.data['nspec']-1, self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=7] dm = np.zeros((self.data['nspec']-1, self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef int i, j, k, m, p, q

		# Extrapolate lnPI
		if (not override):
			# If histogram does not extend far enough, cannot calculate average quantities needed for extrapolation accurately
			assert (np.max(self.data['ln(PI)']) - cutoff > self.data['ln(PI)'][len(self.data['ln(PI)'])-1]), 'Error, histogram edge effect encountered in temperature extrapolation'

		try:
			dlnPI, dm = self._dBMU(skip_mom)
		except:
			raise Exception ('Unable to compute first derivative')

		self.data['ln(PI)'] += dB*dlnPI[0]
		for q in xrange(1,self.data['nspec']):
			self.data['ln(PI)'] += target_dDmu[q-1]*dlnPI[q]

		for i in xrange(self.data['nspec']):
			for j in xrange(self.data['max_order']+1):
				for k in xrange(self.data['nspec']):
					for m in xrange(self.data['max_order']+1):
						for p in xrange(self.data['max_order']+1):
							self.data['mom'][i,j,k,m,p] += dB*dm[0,i,j,k,m,p]
							for q in xrange(1, self.data['nspec']):
								self.data['mom'][i,j,k,m,p] += target_dDmu[q-1]*dm[q,i,j,k,m,p]

	def _temp_dmu_extrap_2_multi(self, np.ndarray[np.double_t, ndim=1] target_betas, np.ndarray[np.double_t, ndim=2] target_dmus, double cutoff=10.0, override=False, skip_mom=False, bool first_order_mom=False):
		"""
		Extrapolate the histogam in an array of different temperatures and dMus using second order corrections.

		Parameters
		----------
		target_betas : ndarray
			Array of 1/kT of the desired distribution
		target_dmus : ndarray
			Array of [desired difference of chemical potentials of species 2-N, dmu_i = mu_i - mu_1]
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10)
		override : bool
			Override warnings about inaccuracies in lnPI (default=False)
		skip_mom : bool
			Skip extrapolation of moments (default=False)
		first_order_mom : bool
			If True, only use first order extrapolation to extrapolate moments (default=False)

		Returns
		-------
		array
			2D array of histogram clones at each [beta, dMu]

		"""

		cdef double dB
		cdef np.ndarray[np.double_t, ndim=1] target_dDmu, xi = np.zeros(self.data['nspec'], dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=2] dlnPI = np.zeros((self.data['nspec'], self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=3] H_lnPI = np.zeros((self.data['nspec'], self.data['nspec'], self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=8] H_mom = np.zeros((self.data['nspec'], self.data['nspec'], self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=7] dm = np.zeros((self.data['nspec'], self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=2] x = np.zeros((self.data['nspec'], self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef int i, j, k, m, p, q, r

		hists = []

		# Extrapolate lnPI
		if (not override):
			# If histogram does not extend far enough, cannot calculate average quantities needed for extrapolation accurately
			assert (np.max(self.data['ln(PI)']) - cutoff > self.data['ln(PI)'][len(self.data['ln(PI)'])-1]), 'Error, histogram edge effect encountered in temperature extrapolation'

		try:
			# For numerical stability, always normalize lnPI before extrapolating
			cc = copy.deepcopy(self)
			cc.normalize()
			dlnPI, dm = cc._dBMU(skip_mom)
			H_lnPI, H_mom = cc._dBMU2(skip_mom)
		except:
			raise Exception ('Unable to compute derivatives')

		for target_beta in target_betas:
			dB = target_beta - self.data['curr_beta']
			hc = []

			for target_dmu in target_dmus:
				target_dDmu = target_dmu - (self.data['curr_mu'][1:] - self.data['curr_mu'][0])

				try:
					clone = copy.deepcopy(self)

					xi[0] = dB
					xi[1:] = target_dDmu[:]

					clone.data['ln(PI)'] += np.dot(xi,dlnPI)
					for q in xrange(clone.data['nspec']):
						x[q,:] = np.dot(xi, H_lnPI[:,q,:])*xi[q]
					clone.data['ln(PI)'] += 0.5*np.sum(x, axis=0)

					for i in xrange(clone.data['nspec']):
						for j in xrange(clone.data['max_order']+1):
							for k in xrange(clone.data['nspec']):
								for m in xrange(clone.data['max_order']+1):
									for p in xrange(clone.data['max_order']+1):
										for q in xrange(clone.data['nspec']):
											clone.data['mom'][i,j,k,m,p] += xi[q]*dm[q,i,j,k,m,p] # 1st order corrections
											x[q,:] = np.dot(xi, H_mom[:,q,i,j,k,m,p,:])*xi[q] # 2nd order corrections
										if (not first_order_mom):
											clone.data['mom'][i,j,k,m,p] += 0.5*np.sum(x, axis=0)

				except:
					hc.append(None)
				else:
					hc.append(clone)

			hists.append(hc)

		return hists

	def _temp_dmu_extrap_2(self, double target_beta, np.ndarray[np.double_t, ndim=1] target_dmu, double cutoff=10.0, bool override=False, bool skip_mom=False, bool first_order_mom=False):
		"""
		Extrapolate the histogam in temperature and dMu using second order corrections.

		Parameters
		----------
		target_beta : double
			1/kT of the desired distribution
		target_dmu : ndarray
			Desired difference of chemical potentials of species 2-N, dmu_i = mu_i - mu_1
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10)
		override : bool
			Override warnings about inaccuracies in lnPI (default=False)
		skip_mom : bool
			Skip extrapolation of moments (default=False)
		first_order_mom : bool
			If True, only use first order extrapolation to extrapolate moments (default=False)

		"""

		cdef double dB = target_beta - self.data['curr_beta']
		cdef np.ndarray[np.double_t, ndim=1] target_dDmu = target_dmu - (self.data['curr_mu'][1:] - self.data['curr_mu'][0]), xi = np.zeros(self.data['nspec'], dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=2] dlnPI = np.zeros((self.data['nspec'], self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=3] H_lnPI = np.zeros((self.data['nspec'], self.data['nspec'], self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=8] H_mom = np.zeros((self.data['nspec'], self.data['nspec'], self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=7] dm = np.zeros((self.data['nspec'], self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=2] x = np.zeros((self.data['nspec'], self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef int i, j, k, m, p, q, r

		# Extrapolate lnPI
		if (not override):
			# If histogram does not extend far enough, cannot calculate average quantities needed for extrapolation accurately
			assert (np.max(self.data['ln(PI)']) - cutoff > self.data['ln(PI)'][len(self.data['ln(PI)'])-1]), 'Error, histogram edge effect encountered in temperature extrapolation'

		try:
			dlnPI, dm = self._dBMU(skip_mom)
			H_lnPI, H_mom = self._dBMU2(skip_mom)
		except:
			raise Exception ('Unable to compure derivatives')

		xi[0] = dB
		xi[1:] = target_dDmu[:]
		self.data['ln(PI)'] += np.dot(xi,dlnPI)
		for q in xrange(self.data['nspec']):
			x[q,:] = np.dot(xi, H_lnPI[:,q,:])*xi[q]
		self.data['ln(PI)'] += 0.5*np.sum(x, axis=0)

		for i in xrange(self.data['nspec']):
			for j in xrange(self.data['max_order']+1):
				for k in xrange(self.data['nspec']):
					for m in xrange(self.data['max_order']+1):
						for p in xrange(self.data['max_order']+1):
							for q in xrange(self.data['nspec']):
								self.data['mom'][i,j,k,m,p] += xi[q]*dm[q,i,j,k,m,p] # 1st order corrections
								x[q,:] = np.dot(xi, H_mom[:,q,i,j,k,m,p,:])*xi[q] # 2nd order corrections
							if (not first_order_mom):
								self.data['mom'][i,j,k,m,p] += 0.5*np.sum(x, axis=0)

	def _gc_fluct_vv(self, a, b):
		"""
		Compute fluctuation in grand canonical ensemble, f(a,b) = <ab> - <a><b>.

		Parameters
		----------
		a : ndarray
			Extensive property a
		b : ndarray
			Extensive property b

		Returns
		-------
		double
			f(a,b) (scalar)

		"""

		assert (len(a) == len(self.data['ln(PI)'])), 'Bad quantity array'
		assert (len(b) == len(self.data['ln(PI)'])), 'Bad quantity array'
		cdef np.ndarray[np.double_t, ndim=1] prob = np.exp(self.data['ln(PI)'])
		cdef double sum_prob = np.sum(prob)
		return np.sum(a*b*prob)/sum_prob - np.sum(a*prob)/sum_prob*np.sum(b*prob)/sum_prob

	def _gc_fluct_vi(self, a, y_idx):
		"""
		Compute fluctuation in grand canonical ensemble, f(a,b) = <ab> - <a><b>.

		Parameters
		----------
		a : ndarray
			Extensive property a
		y_idx : array
			Indices of moment b, (i,j,k,m,p)

		Returns
		-------
		double
			f(a,b) (scalar)

		"""

		assert (len(a) == len(self.data['ln(PI)'])), 'Bad quantity array'
		assert (len(y_idx) == 5), 'Bad indices'
		cdef np.ndarray[np.double_t, ndim=1] prob = np.exp(self.data['ln(PI)'])
		cdef double sum_prob = np.sum(prob)
		return np.sum(a*self.data['mom'][y_idx[0],y_idx[1],y_idx[2],y_idx[3],y_idx[4]]*prob)/sum_prob - np.sum(a*prob)/sum_prob*np.sum(self.data['mom'][y_idx[0],y_idx[1],y_idx[2],y_idx[3],y_idx[4]]*prob)/sum_prob

	def _gc_fluct_iv(self, y_idx, a):
		"""
		Compute fluctuation in grand canonical ensemble, f(a,b) = <ab> - <a><b>.

		Parameters
		----------
		y_idx : array
			Indices of moment b, (i,j,k,m,p)
		a : ndarray
			Extensive property a

		Returns
		-------
		double
			f(a,b) (scalar)

		"""

		assert (len(a) == len(self.data['ln(PI)'])), 'Bad quantity array'
		assert (len(y_idx) == 5), 'Bad indices'
		cdef np.ndarray[np.double_t, ndim=1] prob = np.exp(self.data['ln(PI)'])
		cdef double sum_prob = np.sum(prob)
		return np.sum(a*self.data['mom'][y_idx[0],y_idx[1],y_idx[2],y_idx[3],y_idx[4]]*prob)/sum_prob - np.sum(a*prob)/sum_prob*np.sum(self.data['mom'][y_idx[0],y_idx[1],y_idx[2],y_idx[3],y_idx[4]]*prob)/sum_prob

	def _gc_fluct_ii(self, x_idx, y_idx):
		"""
		Compute fluctuation in grand canonical ensemble, f(a,b) = <ab> - <a><b>.

		Parameters
		----------
		x_idx : array
			Indices of moment a, (i,j,k,m,p)
		y_idx : array
			Indices of moment b, (i,j,k,m,p)

		Returns
		-------
		double
			f(a,b) (scalar)

		"""

		assert (len(x_idx) == 5), 'Bad indices'
		assert (len(y_idx) == 5), 'Bad indices'
		cdef np.ndarray[np.double_t, ndim=1] prob = np.exp(self.data['ln(PI)'])
		cdef double sum_prob = np.sum(prob)
		z_idx = self._mom_prod(x_idx,y_idx)
		return np.sum(self.data['mom'][z_idx[0],z_idx[1],z_idx[2],z_idx[3],z_idx[4]]*prob)/sum_prob - np.sum(self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]]*prob)/sum_prob*np.sum(self.data['mom'][y_idx[0],y_idx[1],y_idx[2],y_idx[3],y_idx[4]]*prob)/sum_prob

	def _gc_ave_v(self, a):
		"""
		Compute an average property in the grand canonical ensemble, <a>.

		Parameters
		----------
		a : ndarray
			Extensive property

		Returns
		-------
		double
			<a> (scalar)

		"""

		assert (len(a) == len(self.data['ln(PI)'])), 'Bad quantity array'
		cdef np.ndarray[np.double_t, ndim=1] prob = np.exp(self.data['ln(PI)'])
		cdef double sum_prob = np.sum(prob)
		return np.sum(a*prob)/sum_prob

	def _gc_ave_i(self, x_idx):
		"""
		Compute an average property in the grand canonical ensemble, <a>.

		Parameters
		----------
		x_idx : array
			Indices of moment to take average of, (i,j,k,m,p)
		n : int
			Exponent on N_tot (default=0)

		Returns
		-------
		double
			<a> (scalar)

		"""

		assert (len(x_idx) == 5), 'Bad indices'
		cdef np.ndarray[np.double_t, ndim=1] prob = np.exp(self.data['ln(PI)'])
		cdef double sum_prob = np.sum(prob)
		return np.sum(self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]]*prob)/sum_prob

	def _gc_dX_dB(self, x_idx, int n=0):
		"""
		Compute the first derivative of a grand-canonical-averaged quantity with respect to beta.

		Quantity is mom[x_idx]*N_tot^n.  To just get N_tot, x_idx = [0,0,0,0,0] (or at least where all exponents are 0). Accounts for KE correction if necessary.

		Parameters
		----------
		x_idx :array
			Indices of moment (i,j,k,m,p)
		n : int
			Exponent on N_tot (default=0)

		Returns
		-------
		double
			derivative (scalar)

		"""

		assert (len(x_idx) == 5), 'Bad indices'
		cdef np.ndarray[np.double_t, ndim=1] X = self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]]*(self.data['ntot']**n), RUN
		cdef double der = 0.0, ave_RUN = 0.0
		cdef int i

		der += self.data['curr_mu'][0]*self._gc_fluct_vv(X, self.data['ntot'])
		der -= self._gc_fluct_vi(X,[0,0,0,0,1])
		for i in xrange(self.data['nspec']):
			der += (self.data['curr_mu'][i]-self.data['curr_mu'][0])*self._gc_fluct_vi(X,[i,1,0,0,0])

		if (self.metadata['used_ke']):
			if (x_idx[4] > 0):
				RUN = self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]-1]*(self.data['ntot']**(n+1))
				ave_RUN = np.sum(np.exp(self.data['ln(PI)'])*RUN)/np.sum(np.exp(self.data['ln(PI)']))
				der -= 1.5*x_idx[4]/(self.data['curr_beta']*self.data['curr_beta'])*ave_RUN

		return der

	def _gc_d2X_dB2(self, x_idx, int n=0):
		"""
		Compute the second derivative of a grand-canonical-averaged quantity with respect to beta.

		Quantity is considered to be X = mom[x_idx]*N_tot^n. Accounts for KE correction if necessary.

		Parameters
		----------
		x_idx : array
			Indices of moment (i,j,k,m,p)
		n : int
			Exponent on N_tot (default=0)

		Returns
		-------
		double
			derivative (scalar)

		"""

		assert (len(x_idx) == 5), 'Bad indices'
		cdef np.ndarray[np.double_t, ndim=1] X = self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]]*(self.data['ntot']**n)
		cdef double der = 0.0, ave_RUN = 0.0, a = 0.0, b = 0.0
		cdef int i

		der = self.data['curr_mu'][0]*self._gc_df_dB_in((x_idx, n), 1) - self._gc_df_dB_ii((x_idx,n),([0,0,0,0,1],0))
		for i in xrange(self.data['nspec']):
			der += (self.data['curr_mu'][i]-self.data['curr_mu'][0])*self._gc_df_dB_ii((x_idx,n),([i,1,0,0,0],0))

		if (self.metadata['used_ke']):
			if (x_idx[4] > 0):
				y_idx = copy.copy(x_idx)
				y_idx[4] -= 1

				ave_RUN = np.sum(np.exp(self.data['ln(PI)'])*self.data['mom'][y_idx[0],y_idx[1],y_idx[2],y_idx[3],y_idx[4]]*(self.data['ntot']**(n+1)))/np.sum(np.exp(self.data['ln(PI)']))
				a = -2.0/self.data['curr_beta']*ave_RUN
				b = self._gc_dX_dB(y_idx,n+1)
				der -= 1.5*x_idx[4]/(self.data['curr_beta']*self.data['curr_beta'])*(a+b)

		return der

	def _gc_df_dB_ii(self, x_idx_t, y_idx_t):
		"""
		Compute the first derivative of a grand-canonical-averaged fluctuation with respect to beta, i.e. d/dB of f(<x>,<y>).

		This is intended for taking the products of moments. Includes corrections if KE contributions were present in the raw data. Quantities are considered as, e.g., mom[x_idx]*N_tot^nx.

		Parameters
		----------
		x_idx_t : tuple
			(Indices of moment to take derivative of, (i,j,k,m,p), nx)
		y_idx_t : tuple
			(Indices of moment to take derivative of, (i,j,k,m,p), ny)

		Returns
		-------
		double
			derivative (scalar)

		"""

		x_idx, nx = x_idx_t
		y_idx, ny = y_idx_t
		z_idx = self._mom_prod(x_idx,y_idx) # unaffected by Ntot powers
		X = self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]]*(self.data['ntot']**nx)
		Y = self.data['mom'][y_idx[0],y_idx[1],y_idx[2],y_idx[3],y_idx[4]]*(self.data['ntot']**ny)
		return self._gc_dX_dB(z_idx,nx+ny) - self._gc_ave_v(X)*self._gc_dX_dB(y_idx,ny) - self._gc_ave_v(Y)*self._gc_dX_dB(x_idx,nx)

	def _gc_df_dB_in(self, x_idx_t, int n=0):
		"""
		Compute the first derivative of a grand-canonical-averaged fluctuation with respect to beta, i.e. d/dB of f(<x>,<y>), where x = (mom[x_idx]*N_tot^nx), and y = N_tot^n.

		Parameters
		----------
		x_idx_t : tuple
			(Indices of moment to take derivative of, (i,j,k,m,p), nx)
		n : int
			Exponent on N_tot (default=0)

		Returns
		-------
		double
			derivative (scalar)

		"""

		cdef np.ndarray[np.double_t, ndim=1] X, Y
		cdef int nx = 0

		x_idx, nx = x_idx_t
		X = self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]]*(self.data['ntot']**nx)
		Y = self.data['mom'][0,0,0,0,0]*(self.data['ntot']**n)

		return self._gc_dX_dB(x_idx,n+nx) - self._gc_ave_v(X)*self._gc_dX_dB([0,0,0,0,0],n) - self._gc_ave_v(Y)*self._gc_dX_dB(x_idx,nx)

	def _order_mom_address(self, idx):
		"""
		Order the address of a moment by particle indices starting from the lowest to the highest.

		e.g. N1^j*N2^m*U^p instead of N2^j*N1^m*U^p

		Parameters
		----------
		idx : array
			Address, i.e., [i,j,k,m,p]

		Returns
		-------
		array
			ordered address

		"""

		assert (len(idx) == 5), 'Bad indices, cannot reorder'
		ord = np.zeros(5, dtype=np.int)
		if (idx[0] > idx[2]):
			ord[0] = idx[2]
			ord[1] = idx[3]
			ord[2] = idx[0]
			ord[3] = idx[1]
			ord[4] = idx[4] # energy power is unaffected
		else:
			ord = copy.copy(idx)

		return ord

	def _mom_prod(self, x_idx, y_idx):
		"""
		Compute the indices for a moment which is the product of individual moments.

		Only valid for pure and binary mixtures, will throw exception otherwise.

		Parameters
		----------
		x_idx : array
			Indices of first moment, X, (i,j,k,m,p)
		y_idx : array
			Indices of second moment, Y, (i,j,k,m,p)

		Returns
		-------
		array
			indices of XY

		"""

		assert (self.data['nspec'] <= 2), 'Ordering moment indices is only valid for 2 or less components'
		z_ord = np.zeros(5, dtype=np.int)
		cdef int diff = 0

		# x,y can be N1N1, N1N2, or N2N2 at this point
		x_ord = copy.copy(x_idx)
		if (x_ord[0] == x_ord[2]):
			# Nx^jNx^m --> Nx^{j+m}Nx^0
			x_ord[1] = x_ord[1] + x_ord[3]
			x_ord[3] = 0
			x_ord[2] = 0 # since exponent is now zero, free to change this
		x_ord = self._order_mom_address(x_ord)

		y_ord = copy.copy(y_idx)
		if (y_ord[0] == y_ord[2]):
			# Nx^jNx^m --> Nx^{j+m}Nx^0
			y_ord[1] = y_ord[1] + y_ord[3]
			y_ord[3] = 0
			y_ord[2] = 0 # since exponent is now zero, free to change this
		y_ord = self._order_mom_address(y_ord)

		# After ordering the reference to particle indices and resetting second component
		# to species 1 with exponent of 0, now x and y can be N1N1, or N1N2
		if (x_ord[0] == y_ord[0] and x_ord[2] == y_ord[2]):
			# if both x and y refer to same things, just add powers
			# both are N1N1, N2N2, or N1N2
			z_ord = copy.copy(x_ord)
			z_ord[1] = z_ord[1]+y_ord[1]
			z_ord[3] = z_ord[3]+y_ord[3]
			z_ord[4] = z_ord[4]+y_ord[4]
		else:
			if (x_ord[0] == 0 and x_ord[2] == 0 and y_ord[0] == 0 and y_ord[2] == 1):
				# x is N1N1 and y is N1N2
				z_ord = copy.copy(y_ord)
				z_ord[1] = z_ord[1]+(x_ord[1]+x_ord[3])
				z_ord[4] = z_ord[4]+x_ord[4]
			elif ():
				# x is N1N1 and y is N2N2
				z_ord = copy.copy(y_ord)
				z_ord[0] = x_ord[0]
				z_ord[1] = x_ord[1]+x_ord[3]
				z_ord[2] = y_ord[0]
				z_ord[3] = y_ord[1]+y_ord[3]
				z_ord[4] = x_ord[4]+y_ord[4]
			elif (x_ord[0] == 0 and x_ord[2] == 1 and y_ord[0] == 0 and y_ord[2] == 0):
				# x is N1N2 and y is N1N1
				z_ord = copy.copy(x_ord)
				z_ord[1] = z_ord[1]+(y_ord[1]+y_ord[3])
				z_ord[4] = z_ord[4]+y_ord[4]
			elif ():
				# x is N1N2 and y is N2N2
				z_ord = copy.copy(y_ord)
				z_ord[0] = x_ord[0]
				z_ord[1] = x_ord[1]
				z_ord[2] = x_ord[2]
				z_ord[3] = x_ord[3]+y_ord[1]+y_ord[3]
				z_ord[4] = x_ord[4]+y_ord[4]
			elif ():
				# x is N2N2 and y is N1N1
				z_ord = copy.copy(y_ord)
				z_ord[0] = y_ord[0]
				z_ord[1] = y_ord[1]+y_ord[3]
				z_ord[2] = x_ord[0]
				z_ord[3] = x_ord[1]+x_ord[3]
				z_ord[4] = x_ord[4]+y_ord[4]
			elif ():
				# x is N2N2 and y is N1N2
				z_ord = copy.copy(y_ord)
				z_ord[0] = y_ord[0]
				z_ord[1] = y_ord[1]
				z_ord[2] = y_ord[2]
				z_ord[3] = x_ord[1]+x_ord[3]+y_ord[3]
				z_ord[4] = x_ord[4]+y_ord[4]
			else:
				raise Exception ('Bad logic')

		# Use symmetry to prevent overflowing max_order
		if (z_ord[0] == z_ord[2]):
			if (z_ord[1] > self.data['max_order']):
				diff = z_ord[1] - self.data['max_order']
				z_ord[1] = self.data['max_order']
				z_ord[3] = diff
			elif (z_ord[3] > self.data['max_order']):
				diff = z_ord[3] - self.data['max_order']
				z_ord[3] = self.data['max_order']
				z_ord[1] = diff

		# ensure both powers were not out of range
		assert (z_ord[1] <= self.data['max_order']), 'Order out of range'
		assert (z_ord[3] <= self.data['max_order']), 'Order out of range'
		assert (z_ord[4] <= self.data['max_order']), 'Order out of range'

		return z_ord

	def _sg_dX_dB(self, x_idx, int n=0):
		"""
		Compute the first derivative of a semi-grand-averaged quantity with respect to beta.

		Treats X = mom[x_idx]*N_tot^n. Accounts for KE corrections if necessary.

		Parameters
		----------
		x_idx : array
			Indices of moment to take derivative of, [i,j,k,m,p]
		n : int
			Exponent on N_tot (default=0)

		Returns
		-------
		ndarray
			derivative

		"""

		assert (len(x_idx) == 5), 'Bad indices'
		cdef np.ndarray[np.double_t, ndim=1] der = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] RU = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] f_XU = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] f_XNi = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] XNi = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef int i

		# all to zero power implies no change
		if (np.all([x_idx[1], x_idx[3], x_idx[4]] == [0,0,0])):
			return der

		if (x_idx[4] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')
		if (x_idx[3] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')
		if (x_idx[1] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')
		f_XU = self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]+1]*(self.data['ntot']**n) - self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]]*(self.data['ntot']**n)*(self.data['mom'][0,0,0,0,1])

		der -= f_XU
		for i in xrange(self.data['nspec']):
			if (x_idx[0] == i and x_idx[1]+1 <= self.data['max_order']):
				XNi = self.data['mom'][x_idx[0],x_idx[1]+1,x_idx[2],x_idx[3],x_idx[4]]*(self.data['ntot']**n)
			elif (x_idx[2] == i and x_idx[3]+1 <= self.data['max_order']):
				XNi = self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3]+1,x_idx[4]]*(self.data['ntot']**n)
			elif (x_idx[1] == 0):
				XNi = self.data['mom'][i,1,x_idx[2],x_idx[3],x_idx[4]]*(self.data['ntot']**n)
			elif (x_idx[3] == 0):
				XNi = self.data['mom'][x_idx[0],x_idx[1],i,1,x_idx[4]]*(self.data['ntot']**n)
			elif (x_idx[0] == x_idx[2] and (x_idx[1]+x_idx[3] <= self.data['max_order'])):
				XNi = self.data['mom'][x_idx[0],x_idx[1]+x_idx[3],i,1,x_idx[4]]*(self.data['ntot']**n)
			else:
				raise Exception ('max_order too low to take this derivative')
			f_XNi = XNi - self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]]*(self.data['ntot']**n)*self.data['mom'][i,1,0,0,0]
			der += (self.data['curr_mu'][i] - self.data['curr_mu'][0])*f_XNi

		if (self.metadata['used_ke']):
			if (x_idx[4] > 0):
				RU = self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]-1]*(self.data['ntot']**n)
				der -= 1.5*x_idx[4]/(self.data['curr_beta']*self.data['curr_beta'])*self.data['ntot']*RU

		return der

	def _sg_dX_dMU(self, int q, x_idx):
		"""
		Compute the first derivative of a semi-grand-averaged quantity with respect to dMu.

		Parameters
		----------
		q : int
			Index of dMu to take derivative with respect to (0 <= q < nspec-1, adjusted for absence of species 1)
		x_idx : array
			Indices of moment to take derivative of, [i,j,k,m,p]

		Returns
		-------
		ndarray
			derivative

		"""

		assert (len(x_idx) == 5), 'Bad indices'
		assert (q >= 0 and q < self.data['nspec']-1), 'Bad dMu index'
		cdef np.ndarray[np.double_t, ndim=1] der = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] XNi = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef int i = q+1

		# all to zero power implies no change
		if (np.all([x_idx[1], x_idx[3], x_idx[4]] == [0,0,0])):
			return der

		if (x_idx[4] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')
		if (x_idx[3] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')
		if (x_idx[1] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')

		if (x_idx[0] == i and x_idx[1]+1 <= self.data['max_order']):
			XNi = self.data['mom'][x_idx[0],x_idx[1]+1,x_idx[2],x_idx[3],x_idx[4]]
		elif (x_idx[2] == i and x_idx[3]+1 <= self.data['max_order']):
			XNi = self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3]+1,x_idx[4]]
		elif (x_idx[1] == 0):
			XNi = self.data['mom'][i,1,x_idx[2],x_idx[3],x_idx[4]]
		elif (x_idx[3] == 0):
			XNi = self.data['mom'][x_idx[0],x_idx[1],i,1,x_idx[4]]
		elif (x_idx[0] == x_idx[2] and (x_idx[1]+x_idx[3] <= self.data['max_order'])):
			XNi = self.data['mom'][x_idx[0],x_idx[1]+x_idx[3],i,1,x_idx[4]]
		else:
			raise Exception ('max_order too low to take this derivative')

		der = self.data['curr_beta']*(XNi - self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]]*self.data['mom'][i,1,0,0,0])

		return der

	def _sg_d2X_dB2(self, x_idx, int n=0):
		"""
		Compute the second derivative of a semi-grand-averaged quantity with respect to beta.

		Treats X = mom[x_idx]*N_tot^n. Accounts for KE corrections if necessary.

		Parameters
		----------
		x_idx : array
			Indices of moment to take derivative of, [i,j,k,m,p]
		n : int
			Exponent on N_tot (default=0)

		Returns
		-------
		ndarray
			derivative

		"""

		assert (len(x_idx) == 5), 'Bad indices'
		cdef np.ndarray[np.double_t, ndim=1] der = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] RU = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] b = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef int i

		# all to zero power implies no change
		if (np.all([x_idx[1], x_idx[3], x_idx[4]] == [0,0,0])):
			return der

		if (x_idx[4] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')
		if (x_idx[3] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')
		if (x_idx[1] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')

		der -= self._sg_df_dB((x_idx,n),([0,0,0,0,1],0))
		for i in xrange(self.data['nspec']):
			der += (self.data['curr_mu'][i] - self.data['curr_mu'][0])*self._sg_df_dB((x_idx,n),([i,1,0,0,0],0))

		if (self.metadata['used_ke']):
			if (x_idx[4] > 0):
				y_idx = copy.copy(x_idx)
				y_idx[4] -= 1
				RU = self.data['mom'][y_idx[0],y_idx[1],y_idx[2],y_idx[3],y_idx[4]]*(self.data['ntot']**n)
				a = -2.0/self.data['curr_beta']*RU
				b = self._sg_dX_dB(y_idx,n)
				der += -1.5*x_idx[4]*self.data['ntot']/(self.data['curr_beta']*self.data['curr_beta'])*(a+b)

		return der

	def _sg_d2X_dMU2(self, int q, int r, x_idx):
		"""
		Compute the second derivative of a semi-grand-averaged quantity with respect to dMu_idMu_j.

		Parameters
		----------
		q : int
			Index of dMu_i to take derivative with respect to (i-1, adjusted for absence of species 1)
		r : int
			Index of dMu_j to take derivative with respect to (j-1, adjusted for absence of species 1)
		x_idx : array
			Indices of moment to take derivative of, [i,j,k,m,p]

		Returns
		-------
		ndarray
			derivative

		"""

		cdef np.ndarray[np.double_t, ndim=1] der = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)

		assert (len(x_idx) == 5), 'Bad indices'
		assert (q >= 0 and q < self.data['nspec']-1), 'Bad dMu index'
		assert (r >= 0 and r < self.data['nspec']-1), 'Bad dMu index'

		# all to zero power implies no change
		if (np.all([x_idx[1], x_idx[3], x_idx[4]] == [0,0,0])):
			return der

		if (x_idx[4] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')
		if (x_idx[3] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')
		if (x_idx[1] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')

		der = self.data['curr_beta']*self._sg_df_dMU(q,x_idx,[r+1,1,0,0,0])

		return der

	def _sg_d3X_dB3(self, x_idx, int n=0):
		"""
		Compute the third derivative of a semi-grand-averaged quantity with respect to beta.

		Treats X = mom[x_idx]*N_tot^n. KE corrections have not been implemented yet, so exception is raised if attempted.

		Parameters
		----------
		x_idx : array
			Indices of moment to take derivative of, [i,j,k,m,p]
		n : int
			Exponent on N_tot (default=0)

		Returns
		-------
		ndarray
			derivative

		"""

		assert (len(x_idx) == 5), 'Bad indices'
		cdef np.ndarray[np.double_t, ndim=1] der = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef int i

		# all to zero power implies no change
		if (np.all([x_idx[1], x_idx[3], x_idx[4]] == [0,0,0])):
			return der

		if (x_idx[4] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')
		if (x_idx[3] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')
		if (x_idx[1] >= self.data['max_order']):
			raise Exception ('max_order too low to take this derivative')

		der -= self._sg_d2f_dB2((x_idx,n),([0,0,0,0,1],0))
		for i in xrange(self.data['nspec']):
			der += (self.data['curr_mu'][i] - self.data['curr_mu'][0])*self._sg_d2f_dB2((x_idx,n),([i,1,0,0,0],0))

		if (self.metadata['used_ke']):
			raise Exception ('No KE correction implemented for _sg_d3X_dB3()')

		return der

	def _sg_df_dB(self, x_idx_t, y_idx_t):
		"""
		Compute the first derivative of a semi-grand-averaged fluctuation with respect to beta, i.e. d/dB of f(x,y), where x = mom[x_idx]*N_tot^nx, y = mom[y_idx]*N_tot^ny.

		Accounts for KE corrections if necessary.

		Parameters
		----------
		x_idx_t : tuple
			(Indices of moment to take derivative of, (i,j,k,m,p), nx)
		y_idx_t : tuple
			(Indices of moment to take derivative of, (i,j,k,m,p), ny)

		Returns
		-------
		ndarray
			derivative

		"""

		x_idx, nx = x_idx_t
		y_idx, ny = y_idx_t
		cdef np.ndarray[np.double_t, ndim=1] der = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		z_idx = self._mom_prod(x_idx, y_idx)

		der = self._sg_dX_dB(z_idx,nx+ny) - self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]]*(self.data['ntot']**nx)*self._sg_dX_dB(y_idx,ny) - self.data['mom'][y_idx[0],y_idx[1],y_idx[2],y_idx[3],y_idx[4]]*(self.data['ntot']**ny)*self._sg_dX_dB(x_idx,nx)

		return der

	def _sg_df_dMU(self, int j, x_idx, y_idx):
		"""
		Compute the first derivative of a semi-grand-averaged fluctuation with respect to dMu_j of f(x_idx, y_idx).

		Parameters
		----------
		j : int
			Species index to take derivative with respect to (0 <= j < nspec-1, adjusted for absence of species 1)
		x_idx : array
			Indices of moment in fluctuation
		y_idx : array
			Indices of moment in fluctuation

		"""

		assert (len(x_idx) == 5), 'Bad indices'
		assert (len(y_idx) == 5), 'Bad indices'
		assert (j >= 0 and j < self.data['nspec']-1), 'Bad species index'
		cdef np.ndarray[np.double_t, ndim=1] der = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		z_idx = self._mom_prod(x_idx,y_idx)

		der = self._sg_dX_dMU(j,z_idx) - self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]]*self._sg_dX_dMU(j,y_idx) - self.data['mom'][y_idx[0],y_idx[1],y_idx[2],y_idx[3],y_idx[4]]*self._sg_dX_dMU(j,x_idx)

		return der

	def _sg_d2f_dB2(self, x_idx_t, y_idx_t):
		"""
		Compute the second derivative of a semi-grand-averaged fluctuation with respect to beta, i.e. d/dB of f(x,y), where x = mom[x_idx]*N_tot^nx, y = mom[y_idx]*N_tot^ny.

		Accounts for KE corrections if necessary.

		Parameters
		----------
		x_idx_t : tuple
			(Indices of moment to take derivative of, (i,j,k,m,p), nx)
		y_idx_t : tuple
			(Indices of moment to take derivative of, (i,j,k,m,p), ny)

		Returns
		-------
		ndarray
			derivative

		"""

		x_idx, nx = x_idx_t
		y_idx, ny = y_idx_t
		cdef np.ndarray[np.double_t, ndim=1] der = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		z_idx = self._mom_prod(x_idx, y_idx)

		return self._sg_d2X_dB2(z_idx,nx+ny) - self.data['mom'][x_idx[0],x_idx[1],x_idx[2],x_idx[3],x_idx[4]]*(self.data['ntot']**nx)*self._sg_d2X_dB2(y_idx,ny) - self._sg_dX_dB(x_idx,nx)*self._sg_dX_dB(y_idx,ny) - self.data['mom'][y_idx[0],y_idx[1],y_idx[2],y_idx[3],y_idx[4]]*(self.data['ntot']**ny)*self._sg_d2X_dB2(x_idx,nx) - self._sg_dX_dB(x_idx,nx)*self._sg_dX_dB(y_idx,ny)

	def _temp_extrap_1(self, double target_beta, double cutoff=10.0, override=False, skip_mom=False):
		"""
		First order temperature extrapolation on self.

		Only extrapolates ln(PI) and <X> properties, where X is some extensive property stored in the moments vectors. Higher order moments are ignored for now.

		Parameters
		----------
		target_beta : double
			1/kT of the desired distribution
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10), value of 0 will deactivate
		override : bool
			Override warnings about inaccuracies in lnPI due to histogram edge effects (default=False)
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		"""

		cdef double dB = target_beta - self.data['curr_beta']
		cdef np.ndarray[np.double_t, ndim=1] dlnPI_dB = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=6] dm_dB = np.zeros((self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)

		# extrapolate lnPI
		if (not override):
			# if histogram does not extend far enough, cannot calculate average quantities needed for extrapolation accurately
			assert (np.max(self.data['ln(PI)']) - cutoff > self.data['ln(PI)'][len(self.data['ln(PI)'])-1]), 'Error, histogram edge effect encountered in temperature extrapolation'

		dlnPI_dB, dm_dB = self._dB(skip_mom)

		self.data['ln(PI)'] += dB*dlnPI_dB
		for i in xrange(self.data['nspec']):
			for j in xrange(self.data['max_order']+1):
				for k in xrange(self.data['nspec']):
					for m in xrange(self.data['max_order']+1):
						for p in xrange(self.data['max_order']+1):
							self.data['mom'][i,j,k,m,p] += dB*dm_dB[i,j,k,m,p]

	def _temp_extrap_2(self, double target_beta, double cutoff=10.0, override=False, skip_mom=False):
		"""
		Second order temperature extrapolation on self.

		Parameters
		----------
		target_beta : double
			1/kT of the desired distribution
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10), value of 0 will deactivate
		override : bool
			Override warnings about inaccuracies in lnPI due to histogram edge effects (default=False)
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		"""

		cdef double dB = target_beta - self.data['curr_beta']
		cdef np.ndarray[np.double_t, ndim=1] dlnPI_dB = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] d2lnPI_dB2 = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=6] dm_dB = np.zeros((self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=6] d2m_dB2 = np.zeros((self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)

		# extrapolate lnPI
		if (not override):
			# if histogram does not extend far enough, cannot calculate average quantities needed for extrapolation accurately
			assert (np.max(self.data['ln(PI)']) - cutoff > self.data['ln(PI)'][len(self.data['ln(PI)'])-1]), 'Error, histogram edge effect encountered in temperature extrapolation'

		dlnPI_dB, dm_dB = self._dB(skip_mom)
		d2lnPI_dB2, d2m_dB2 = self._dB2(skip_mom)

		self.data['ln(PI)'] += dB*dlnPI_dB + 0.5*dB*dB*d2lnPI_dB2
		for i in xrange(self.data['nspec']):
			for j in xrange(self.data['max_order']+1):
				for k in xrange(self.data['nspec']):
					for m in xrange(self.data['max_order']+1):
						for p in xrange(self.data['max_order']+1):
							self.data['mom'][i,j,k,m,p] += dB*dm_dB[i,j,k,m,p] + 0.5*dB*dB*d2m_dB2[i,j,k,m,p]

	def _temp_extrap_3(self, double target_beta, double cutoff=10.0, override=False, skip_mom=False):
		"""
		Third order temperature extrapolation on self.

		Parameters
		----------
		target_beta : double
			1/kT of the desired distribution
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10), value of 0 will deactivate
		override : bool
			Override warnings about inaccuracies in lnPI due to histogram edge effects (default=False)
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		"""

		cdef double dB = target_beta - self.data['curr_beta']
		cdef np.ndarray[np.double_t, ndim=1] dlnPI_dB = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] d2lnPI_dB2 = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] d3lnPI_dB3 = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=6] dm_dB = np.zeros((self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=6] d2m_dB2 = np.zeros((self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=6] d3m_dB3 = np.zeros((self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)

		# extrapolate lnPI
		if (not override):
			# if histogram does not extend far enough, cannot calculate average quantities needed for extrapolation accurately
			assert (np.max(self.data['ln(PI)']) - cutoff > self.data['ln(PI)'][len(self.data['ln(PI)'])-1]), 'Error, histogram edge effect encountered in temperature extrapolation'

		dlnPI_dB, dm_dB = self._dB(skip_mom)
		d2lnPI_dB2, d2m_dB2 = self._dB2(skip_mom)
		d3lnPI_dB3, d3m_dB3 = self._dB3(skip_mom)

		self.data['ln(PI)'] += dB*dlnPI_dB + 0.5*dB*dB*d2lnPI_dB2 + (1.0/6.0)*dB*dB*dB*d3lnPI_dB3
		for i in xrange(self.data['nspec']):
			for j in xrange(self.data['max_order']+1):
				for k in xrange(self.data['nspec']):
					for m in xrange(self.data['max_order']+1):
						for p in xrange(self.data['max_order']+1):
							self.data['mom'][i,j,k,m,p] += dB*dm_dB[i,j,k,m,p] + 0.5*dB*dB*d2m_dB2[i,j,k,m,p] + (1.0/6.0)*dB*dB*dB*d3m_dB3[i,j,k,m,p]

	def _dB(self, skip_mom=False):
		"""
		Calculate first order corrections to properties and lnPI for temperature extrapolation.

		Cannot compute changes for anything of order = max_order so these quantities have 0 change.

		Parameters
		----------
		skip_mom : double
			Skip extrapolation of moments (default=False)

		Returns
		-------
		ndarray, ndarray
			dlnPI_dB, dm_dB

		"""

		cdef int i, j, k, m, p
		cdef double sum_prob, ave_ntot = 0.0, ave_u = 0.0
		cdef np.ndarray[np.double_t, ndim=1] prob, ave_n = np.zeros(self.data['nspec'], dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] dlnPI_dB = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=6] dm_dB = np.zeros((self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		prob = np.exp(self.data['ln(PI)'])
		sum_prob = np.sum(prob)

		# average only the necessary moments at current conditions (up to first power)
		ave_u = np.sum(prob*self.data['mom'][0,0,0,0,1])/sum_prob
		for i in xrange(self.data['nspec']):
			ave_n[i] = np.sum(prob*self.data['mom'][i,1,0,0,0])/sum_prob
			ave_ntot += ave_n[i]

		for i in xrange(self.data['nspec']):
			dlnPI_dB += (self.data['curr_mu'][i] - self.data['curr_mu'][0])*(self.data['mom'][i,1,0,0,0] - ave_n[i])
		dlnPI_dB += self.data['curr_mu'][0]*(self.data['ntot'] - ave_ntot)
		dlnPI_dB -= (self.data['mom'][0,0,0,0,1] - ave_u)

		if (not skip_mom):
			for i in xrange(self.data['nspec']):
				for j in xrange(self.data['max_order']+1):
					for k in xrange(self.data['nspec']):
						for m in xrange(self.data['max_order']+1):
							for p in xrange(self.data['max_order']+1):
								if (j+m+p+1 <= self.data['max_order']):
									try:
										x = self._sg_dX_dB([i,j,k,m,p],0)
									except Exception as e:
										raise Exception ('Cannot compute first derivative: '+str(e))
									else:
										dm_dB[i,j,k,m,p,:] = copy.copy(x)

		return dlnPI_dB, dm_dB

	def _dB2(self, skip_mom=False):
		"""
		Calculate second order corrections to properties and lnPI for temperature extrapolation.

		Parameters
		----------
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		Returns
		-------
		ndarray, ndarray
			d2lnPI_dB2, d2m_dB2

		"""

		cdef int i, j, k, m, p
		cdef np.ndarray[np.double_t, ndim=1] d2lnPI_dB2 = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=6] d2m_dB2 = np.zeros((self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)

		for i in xrange(self.data['nspec']):
			d2lnPI_dB2 += (self.data['curr_mu'][i] - self.data['curr_mu'][0])*(self._sg_dX_dB([i,1,0,0,0],0) - self._gc_dX_dB([i,1,0,0,0],0))
		d2lnPI_dB2 += self.data['curr_mu'][0]*(-self._gc_dX_dB([0,0,0,0,0],1))
		d2lnPI_dB2 -= (self._sg_dX_dB([0,0,0,0,1],0) - self._gc_dX_dB([0,0,0,0,1],0))

		if (not skip_mom):
			for i in xrange(self.data['nspec']):
				for j in xrange(self.data['max_order']+1):
					for k in xrange(self.data['nspec']):
						for m in xrange(self.data['max_order']+1):
							for p in xrange(self.data['max_order']+1):
								if (j+m+p+2 <= self.data['max_order']):
									try:
										x = self._sg_d2X_dB2([i,j,k,m,p],0)
									except Exception as e:
										raise Exception ('Cannot compute second derivative: '+str(e))
									else:
										d2m_dB2[i,j,k,m,p,:] = copy.copy(x)

		return d2lnPI_dB2, d2m_dB2

	def _dB3(self, skip_mom=False):
		"""
		Calculate third order corrections to properties and lnPI for temperature extrapolation.

		This is only valid for binary or pure component systems since for N > 2, 3-particle correlations are necessary. Have not yet implemented 3rd order corrections for presence of KE, so will warn.

		Parameters
		----------
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		Returns
		-------
		ndarray, ndarray
			d3lnPI_dB3, d3m_dB3

		"""

		if (self.metadata['used_ke']):
			raise Exception('KE corrections not implemented for 3rd order beta extrapolation')

		cdef int i, j, k, m, p
		cdef np.ndarray[np.double_t, ndim=1] d3lnPI_dB3 = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=6] d3m_dB3 = np.zeros((self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)

		for i in xrange(self.data['nspec']):
			d3lnPI_dB3 += (self.data['curr_mu'][i] - self.data['curr_mu'][0])*(self._sg_d2X_dB2([i,1,0,0,0],0) - self._gc_d2X_dB2([i,1,0,0,0],0))
		d3lnPI_dB3 += self.data['curr_mu'][0]*(-self._gc_d2X_dB2([0,0,0,0,0],1))
		d3lnPI_dB3 -= (self._sg_d2X_dB2([0,0,0,0,1],0) - self._gc_d2X_dB2([0,0,0,0,1],0))

		if (not skip_mom):
			for i in xrange(self.data['nspec']):
				for j in xrange(self.data['max_order']+1):
					for k in xrange(self.data['nspec']):
						for m in xrange(self.data['max_order']+1):
							for p in xrange(self.data['max_order']+1):
								if (j+m+p+3 <= self.data['max_order']):
									try:
										x = self._sg_d3X_dB3([i,j,k,m,p],0)
									except Exception as e:
										raise Exception ('Cannot compute third derivative: '+str(e))
									else:
										d3m_dB3[i,j,k,m,p,:] = copy.copy(x)

		return d3lnPI_dB3, d3m_dB3

	def _dmu_extrap_1(self, np.ndarray[np.double_t, ndim=1] target_dmu, double cutoff=10.0, override=False, skip_mom=False):
		"""
		Extrapolate the histogram in dMu using first order corrections.

		Parameters
		-----
		target_dDmu : ndarray
			Desired difference of chemical potentials of species 2-N, dmu_i = mu_i - mu_1
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10)
		override : bool
			Override warnings about inaccuracies in lnPI (default=False)
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		"""

		cdef np.ndarray[np.double_t, ndim=1] target_dDmu = target_dmu - (self.data['curr_mu'][1:] - self.data['curr_mu'][0])
		cdef np.ndarray[np.double_t, ndim=2] dlnPI_dDmu = np.zeros((self.data['nspec']-1, self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=7] dm_dDmu = np.zeros((self.data['nspec']-1, self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef int i, j, k, m, p, q

		# extrapolate lnPI
		if (not override):
			# if histogram does not extend far enough, cannot calculate average quantities needed for extrapolation accurately
			assert (np.max(self.data['ln(PI)']) - cutoff > self.data['ln(PI)'][len(self.data['ln(PI)'])-1]), 'Error, histogram edge effect encountered in temperature extrapolation'

		dlnPI_dDmu, dm_dDmu = self._dMU(skip_mom)

		for q in xrange(self.data['nspec']-1):
			self.data['ln(PI)'] += target_dDmu[q]*dlnPI_dDmu[q]

		for i in xrange(self.data['nspec']):
			for j in xrange(self.data['max_order']+1):
				for k in xrange(self.data['nspec']):
					for m in xrange(self.data['max_order']+1):
						for p in xrange(self.data['max_order']+1):
							for q in xrange(self.data['nspec']-1):
								self.data['mom'][i,j,k,m,p] += target_dDmu[q]*dm_dDmu[q,i,j,k,m,p]

	def _dmu_extrap_2(self, np.ndarray[np.double_t, ndim=1] target_dmu, double cutoff=10.0, override=False, skip_mom=False):
		"""
		Extrapolate the histogram in dMu using second order corrections (Hessian matrix).

		Parameters
		----------
		target_dmu : ndarray
			Desired difference of chemical potentials of species 2-N, dmu_i = mu_i - mu_1
		cutoff : double
			Difference in lnPI between maxima and edge to be considered safe to attempt reweighting (default=10)
		override : bool
			Override warnings about inaccuracies in lnPI (default=False)
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		"""

		cdef np.ndarray[np.double_t, ndim=1] target_dDmu = target_dmu - (self.data['curr_mu'][1:] - self.data['curr_mu'][0])
		cdef np.ndarray[np.double_t, ndim=2] dlnPI_dDmu = np.zeros((self.data['nspec']-1, self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=7] dm_dDmu = np.zeros((self.data['nspec']-1, self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=2] x = np.zeros((self.data['nspec']-1, self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef int i, j, k, m, p, q, r

		# extrapolate lnPI
		if (not override):
			# if histogram does not extend far enough, cannot calculate average quantities needed for extrapolation accurately
			assert (np.max(self.data['ln(PI)']) - cutoff > self.data['ln(PI)'][len(self.data['ln(PI)'])-1]), 'Error, histogram edge effect encountered in temperature extrapolation'

		dlnPI_dDmu, dm_dDmu = self._dMU(skip_mom)
		H_lnPI, H_mom = self._dMU2(skip_mom)

		for q in xrange(self.data['nspec']-1):
			self.data['ln(PI)'] += target_dDmu[q]*dlnPI_dDmu[q]

		for q in xrange(self.data['nspec']-1):
			x[q,:] = np.dot(target_dDmu, H_lnPI[:,q,:])*target_dDmu[q]
		self.data['ln(PI)'] += 0.5*np.sum(x, axis=0)

		for i in xrange(self.data['nspec']):
			for j in xrange(self.data['max_order']+1):
				for k in xrange(self.data['nspec']):
					for m in xrange(self.data['max_order']+1):
						for p in xrange(self.data['max_order']+1):
							for q in xrange(self.data['nspec']-1):
								self.data['mom'][i,j,k,m,p] += target_dDmu[q]*dm_dDmu[q,i,j,k,m,p] # First order corrections
								x[q,:] = np.dot(target_dDmu, H_mom[:,q,i,j,k,m,p,:])*target_dDmu[q] # Second order corrections
							self.data['mom'][i,j,k,m,p] += 0.5*np.sum(x, axis=0)

	def _dMU(self, skip_mom=False):
		"""
		Calculate first order corrections to properties and lnPI for dMu extrapolation.

		Cannot compute changes for anything of order = max_order so these quantities have 0 change.

		Parameters
		----------
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		Returns
		-------
		ndarray, ndarray
			dlnPI_dmu, dm_dmu

		"""

		cdef int i, j, k, m, p, q
		cdef double sum_prob
		cdef np.ndarray[np.double_t, ndim=1] prob, ave_n = np.zeros(self.data['nspec']-1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=2] dlnPI_dmu = np.zeros((self.data['nspec']-1, self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=7] dm_dmu = np.zeros((self.data['nspec']-1, self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		prob = np.exp(self.data['ln(PI)'])
		sum_prob = np.sum(prob)

		for i in xrange(self.data['nspec']-1):
			ave_n[i] = np.sum(prob*self.data['mom'][i+1,1,0,0,0])/sum_prob
			dlnPI_dmu[i] = self.data['curr_beta']*(self.data['mom'][i+1,1,0,0,0] - ave_n[i])

		if (not skip_mom):
			for q in xrange(self.data['nspec']-1):
				for i in xrange(self.data['nspec']):
					for j in xrange(self.data['max_order']+1):
						for k in xrange(self.data['nspec']):
							for m in xrange(self.data['max_order']+1):
								for p in xrange(self.data['max_order']+1):
									if (j+m+p+1 <= self.data['max_order']):
										try:
											x = self._sg_dX_dMU(q,[i,j,k,m,p])
										except Exception as e:
											raise Exception ('Cannot compute first derivative: '+str(e))
										else:
											dm_dmu[q,i,j,k,m,p,:] = copy.copy(x)

		return dlnPI_dmu, dm_dmu

	def _dMU2(self, skip_mom=False):
		"""
		Calculate second order corrections (Hessian matrix) to properties and lnPI for dMu extrapolation.

		Cannot compute changes for anything of order = max_order so these quantities have 0 change.

		Parameters
		----------
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		Returns
		-------
		ndarray, ndarray
			H_lnPI, H_mom

		"""

		cdef np.ndarray[np.double_t, ndim=3] H_lnPI = np.zeros((self.data['nspec']-1, self.data['nspec']-1, self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=8] H_mom = np.zeros((self.data['nspec']-1, self.data['nspec']-1, self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] x = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] f = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef int i, j, k, m, p, q, r

		for i in xrange(self.data['nspec']-1):
			for j in xrange(self.data['nspec']-1):
				f = self.data['mom'][i+1,1,j+1,1,0] - self.data['mom'][i+1,1,j+1,0,0]*self.data['mom'][i+1,0,j+1,1,0]
				H_lnPI[i,j] = self.data['curr_beta']**2*(f - self._gc_fluct_ii([i+1,1,0,0,0],[j+1,1,0,0,0]))

		if (not skip_mom):
			for q in xrange(self.data['nspec']-1):
				for r in xrange(self.data['nspec']-1):
					for i in xrange(self.data['nspec']):
						for j in xrange(self.data['max_order']+1):
							for k in xrange(self.data['nspec']):
								for m in xrange(self.data['max_order']+1):
									for p in xrange(self.data['max_order']+1):
										if (j+m+p+2 <= self.data['max_order']):
											try:
												x = self._sg_d2X_dMU2(q,r,[i,j,k,m,p])
											except Exception as e:
												raise Exception ('Cannot compute second derivative: '+str(e))
											else:
												H_mom[q,r,i,j,k,m,p,:] = copy.copy(x)

		return H_lnPI, H_mom

	def _dBMU(self, skip_mom=False):
		"""
		Calculate first order corrections to properties and lnPI for simultaneous dMu and dB extrapolation.

		Cannot compute changes for anything of order = max_order so these quantities have 0 change.

		Parameters
		----------
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		Returns
		-------
		ndarray, ndarray
			dlnPI, dm

		"""

		cdef int i, j, k, m, p, q
		cdef double sum_prob
		cdef np.ndarray[np.double_t, ndim=1] prob, ave_n = np.zeros(self.data['nspec']-1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=2] dlnPI = np.zeros((self.data['nspec'], self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=7] dm = np.zeros((self.data['nspec'], self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		prob = np.exp(self.data['ln(PI)'])
		sum_prob = np.sum(prob)

		dlnPI[0], dm[0,:,:,:,:,:,:] = self._dB(skip_mom)
		for i in xrange(1,self.data['nspec']):
			ave_n[i-1] = np.sum(prob*self.data['mom'][i,1,0,0,0])/sum_prob
			dlnPI[i] = self.data['curr_beta']*(self.data['mom'][i,1,0,0,0] - ave_n[i-1])

		if (not skip_mom):
			for q in xrange(1,self.data['nspec']):
				for i in xrange(self.data['nspec']):
					for j in xrange(self.data['max_order']+1):
						for k in xrange(self.data['nspec']):
							for m in xrange(self.data['max_order']+1):
								for p in xrange(self.data['max_order']+1):
									if (j+m+p+1 <= self.data['max_order']):
										try:
											x = self._sg_dX_dMU(q-1,[i,j,k,m,p])
										except Exception as e:
											raise Exception ('Cannot compute first derivative: '+str(e))
										else:
											dm[q,i,j,k,m,p,:] = copy.copy(x)

		return dlnPI, dm

	def _dBMU2(self, skip_mom=False):
		"""
		Calculate second order corrections (Hessian matrix) to properties and lnPI for simulataneous dMu and dB extrapolation.

		Cannot compute changes for anything of order = max_order so these quantities have 0 change.

		Parameters
		----------
		skip_mom : bool
			Skip extrapolation of moments (default=False)

		Returns
		-------
		ndarray, ndarray
			H_lnPI, H_mom

		"""

		cdef np.ndarray[np.double_t, ndim=3] H_lnPI = np.zeros((self.data['nspec'], self.data['nspec'], self.data['ub']-self.data['lb']+1), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=8] H_mom = np.zeros((self.data['nspec'], self.data['nspec'], self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1, len(self.data['ln(PI)'])), dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] x = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef np.ndarray[np.double_t, ndim=1] f = np.zeros(self.data['ub']-self.data['lb']+1, dtype=np.float64)
		cdef int i, j, k, m, p, q, r

		# get dmu contributions
		for i in xrange(self.data['nspec']-1):
			for j in xrange(self.data['nspec']-1):
				f = self.data['mom'][i+1,1,j+1,1,0] - self.data['mom'][i+1,1,j+1,0,0]*self.data['mom'][i+1,0,j+1,1,0]
				H_lnPI[i+1,j+1,:] = self.data['curr_beta']**2*(f - self._gc_fluct_ii([i+1,1,0,0,0],[j+1,1,0,0,0]))

		if (not skip_mom):
			for q in xrange(self.data['nspec']-1):
				for r in xrange(self.data['nspec']-1):
					for i in xrange(self.data['nspec']):
						for j in xrange(self.data['max_order']+1):
							for k in xrange(self.data['nspec']):
								for m in xrange(self.data['max_order']+1):
									for p in xrange(self.data['max_order']+1):
										if (j+m+p+2 <= self.data['max_order']):
											try:
												x = self._sg_d2X_dMU2(q,r,[i,j,k,m,p])
											except Exception as e:
												raise Exception ('Cannot compute second derivative: '+str(e))
											else:
												H_mom[q+1,r+1,i,j,k,m,p,:] = copy.copy(x)

		# get beta contribution (if skip_mom, returns null matrix for H_mom)
		H_lnPI[0,0], H_mom[0,0] = self._dB2(skip_mom)

		# get beta-dmu contributions
		for q in xrange(1, self.data['nspec']):
			tmp = self.data['mom'][q,1,0,0,0] - np.sum(np.exp(self.data['ln(PI)'])*self.data['mom'][q,1,0,0,0])/np.sum(np.exp(self.data['ln(PI)']))
			tmp += self.data['curr_beta']*(self._sg_dX_dB([q,1,0,0,0],0) - self._gc_dX_dB([q,1,0,0,0],0))

			# symmetry
			H_lnPI[q,0,:] = copy.copy(tmp)
			H_lnPI[0,q,:] = copy.copy(tmp)

		# finish moment Hessian
		if (not skip_mom):
			for q in xrange(1, self.data['nspec']):
				for i in xrange(self.data['nspec']):
					for j in xrange(self.data['max_order']+1):
						for k in xrange(self.data['nspec']):
							for m in xrange(self.data['max_order']+1):
								for p in xrange(self.data['max_order']+1):
									if (j+m+p+2 <= self.data['max_order']):
										try:
											z_idx = self._mom_prod([q,1,0,0,0],[i,j,k,m,p])
											f = self.data['mom'][z_idx[0],z_idx[1],z_idx[2],z_idx[3],z_idx[4]] - self.data['mom'][q,1,0,0,0]*self.data['mom'][i,j,k,m,p]
											x = self.data['curr_beta']*self._sg_df_dB(([q,1,0,0,0],0),([i,j,k,m,p],0)) + f
											#x = self.data['curr_beta']*(self._sg_df_dB(([q,1,0,0,0],0),([i,j,k,m,p],0)) + f) # parantheases should not have been distributed on the last f term
										except Exception as e:
											raise Exception ('Cannot compute second derivative: '+str(e))
										else:
											# symmetry
											H_mom[q,0,i,j,k,m,p,:] = copy.copy(x)
											H_mom[0,q,i,j,k,m,p,:] = copy.copy(x)

		return H_lnPI, H_mom

histogram._cy_normalize = types.MethodType(_cython_normalize, None, histogram)
histogram._cy_reweight = types.MethodType(_cython_reweight, None, histogram)

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double phase_eq_error (double mu_guess, object orig_hist, double beta, np.ndarray[np.double_t, ndim=1] dMu, int order, double cutoff, bool override, int min_width, collect=None):
	"""
	Compute the difference between the area under the lnPI distribution (free energy/kT) for different phases at a given chemical potential of species 1.

	If calculation is requested at a different temperature, reweighting is first done, then temperature extrapolation.

	Parameters
	----------
	mu_guess : double
		Guess for chemical potential of species 1 at coexistence
	orig_hist : histogram
		Histogram to be reweighted in search of coexistence
	beta : double
		Temperature at which to seek equilibrium
	dMu : ndarray
		Desired difference of chemical potentials of species 2-N, dmu_i = mu_i - mu_1
	order : int
		Order of temperature extrapolation to use if beta and dMu are not the same as in hist
	cutoff : double
		Difference in lnPI between maxima and edge to be considered safe to attempt reweighting
	override : bool
		Override warnings about inaccuracies in lnPI after temperature extrapolation
	min_width : int
		Minimum width of a phase to be considered a "real" one instead of background noise
	collect : function (histogram)
		Function that will "collect" maxima and minima into phases if each peak does not represent a phase, which takes a keyword argument "hist" to accept this class.
		This function must modify the class and report a maxima, boudned by minima, for each resulting phase. See example function in collect.py called janus_collect(). (default=None)

	Returns
	-------
	double
		square error in (free energy)/kT between phases

	"""

	# if 99.99% of phase is under one peak, has lnPI = ln(0.9999) = 1e-4, and ln(1-0.9999) = -9.21, so feasible diff ~ 10
	cdef int i, j
	hist = copy.deepcopy(orig_hist)
	hist.reweight(mu_guess)
	curr_dMu = np.array([hist.data['curr_mu'][i] - hist.data['curr_mu'][0] for i in xrange(1, hist.data['nspec'])])
	if (beta != orig_hist.data['curr_beta'] or np.all(curr_dMu == dMu) == False):
		hist.temp_dmu_extrap(beta, dMu, order, cutoff, override, False, True)
	hist.thermo(props=False, collect=collect)

	cdef double default = 100.0 # 10.0**2
	cdef int counter = 0, num_phases = len(hist.data['thermo'])
	cdef np.ndarray[np.double_t, ndim=1] err2_array = np.ones(num_phases*(num_phases-1)/2)*default
	cdef double err2

	if (num_phases == 1):
		err2 = default
	else:
		for i in xrange(num_phases):
			if (hist.data['thermo'][i]['bound_idx'][1] - hist.data['thermo'][i]['bound_idx'][0] >= min_width):
				for j in xrange(i+1, num_phases):
					if (hist.data['thermo'][j]['bound_idx'][1] - hist.data['thermo'][j]['bound_idx'][0] >= min_width):
						err2_array[counter] = (hist.data['thermo'][i]['F.E./kT'] - hist.data['thermo'][j]['F.E./kT'])**2
						counter += 1
		err2 = np.min(err2_array) # min because want to find the phases closest in free energy

	return err2

if __name__ == '__main__':
	print "gc_hist.pyx"

	"""

	* Tutorial:

	To compute the thermodynamic properties of a distribution at the current temperature

	1. Instantiate histogram object from file
	2. Reweight to desired chemical potential (mu1)
	3. Use thermo() to get thermodynamic properties of each phase
	4. Call is_safe() to check that the ln(PI) distribution extends far enough to trust this result

	To compute to another temperature/dMu

	1. Instantiate histogram object from file
	2. Reweight to desired chemical potential (mu1)
	3. Call histogram = temp_extrap() with the appropriate flag set to either modify self or create a copy / use temp_dmu_extrap() if extrapolating in temperature and dMu
	4. histogram.thermo()
		Note that a collection function can be specified to gather peaks into "macrophases" as necessary
	5. histogram.is_safe()

	* Notes:

	Underflows are ignored to allow easy calculation of phase properties.
	The histogram's data is changed after each reweighting operation, but may be continuously reweighted iteratively since it maintains a record of its current conditions.
	The histogram's metadata is static, but its data is modified by operations.

	* To Do:

	Add Butterworth filter
	Check Lowess filter implementation
	Add mixing for energy and particle number histograms in mix()

	"""
