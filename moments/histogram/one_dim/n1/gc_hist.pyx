"""@docstring
@brief Read and manipulate a 1D histogram from flat histogram simulations in grand canonical ensemble where N_1 is the order parameter
@author Nathan A. Mahynski
@date 03/27/2017
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
	Compute the natural logarithm of the sum of a pair of exponentials.  i.e. ln(exp(a) + exp(b))/

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
	Cythonized normalization of histogram (histogram.normalize).

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
	Cythonized normalization of histogram (histogram.reweight).

	"""

	self.data['ln(PI)'] += (mu1_new - self.data['curr_mu'][0])*self.data['curr_beta']*self.data['n1']
	self.normalize()

class histogram (object):
	"""
	Class which reads 1D histogram and computes the thermodynamic properties by reweighting, initialized from a netcdf4 file. This is designed to perform thermodynamic operations (reweighting, etc.) when N_1 is the order parameter from grand canonical ensemble.
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
		self.data['n1'] = np.array(dataset.variables["N_{1}"][:], dtype=np.int)
		self.data['lb'] = self.data['n1'][0]
		self.data['ub'] = self.data['n1'][len(self.data['n1'])-1]
		assert(self.data['lb'] < self.data['ub']), 'Error, bad bounds for N_1'
		self.data['pk_hist'] = {}
		self.data['pk_hist']['hist'] = np.array(dataset.variables["P_{N_i}(N_{1})"][:])
		self.data['pk_hist']['lb'] = np.array(dataset.variables["P_{N_i}(N_{1})_{lb}"][:])
		self.data['pk_hist']['ub'] = np.array(dataset.variables["P_{N_i}(N_{1})_{ub}"][:])
		self.data['pk_hist']['bw'] = np.array(dataset.variables["P_{N_i}(N_{1})_{bw}"][:])
		self.data['e_hist'] = {}
		self.data['e_hist']['hist'] = np.array(dataset.variables["P_{U}(N_{1})"][:])
		self.data['e_hist']['lb'] = np.array(dataset.variables["P_{U}(N_{1})_{lb}"][:])
		self.data['e_hist']['ub'] = np.array(dataset.variables["P_{U}(N_{1})_{ub}"][:])
		self.data['e_hist']['bw'] = np.array(dataset.variables["P_{U}(N_{1})_{bw}"][:])
		self.data['mom'] = np.array(dataset.variables["N_{i}^{j}*N_{k}^{m}*U^{p}"][:])
		assert(self.data['mom'].shape == (self.data['nspec'],self.data['max_order']+1,self.data['nspec'],self.data['max_order']+1,self.data['max_order']+1,len(self.data['n1'])))

		dataset.close()

	def mix(self, other, weights):
		"""
		Create a new histogram that is a "mix" of this one and another one.
		Will only mix histograms at the same temperature, and chemical potentials.
		Properties, X, are mixed such that X(new) = [X(self)*weight_self + X(other)*weight_other]/[weight_self + weight_other].
		This requires that both histograms start from the same lower bound (i.e. N_1 = 0 usually), however
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
		self._cy_reweight(mu1_target)
		self.data['curr_mu'][0] = mu1_target
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

		cdef int last_idx = len(self.data['ln(PI)'])-1

		self.data['ln(PI)_maxima_idx'] = argrelextrema(self.data['ln(PI)'], np.greater, 0, self.metadata['smooth'], 'clip')[0]
		self.data['ln(PI)_minima_idx'] = argrelextrema(self.data['ln(PI)'], np.less, 0, self.metadata['smooth'], 'clip')[0]

		# argrelextrema does NOT include endpoints even if ln(PI) is a "straight" line, so manually include the bounds as well - for now, just compare neighbors
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

		# check that maxima and minima alternate
		assert (np.abs(len(self.data['ln(PI)_maxima_idx']) - len(self.data['ln(PI)_minima_idx'])) <= 1), 'There are '+str(len(self.data['ln(PI)_maxima_idx']))+' local maxima and '+str(len(self.data['ln(PI)_minima_idx']))+' local minima, so cannot be alternating, try adjusting the value of smooth'
		order = np.zeros(len(self.data['ln(PI)_maxima_idx'])+len(self.data['ln(PI)_minima_idx']))
		if (self.data['ln(PI)_maxima_idx'][0] < self.data['ln(PI)_minima_idx'][0]):
			order[::2] = self.data['ln(PI)_maxima_idx']
			order[1::2] = self.data['ln(PI)_minima_idx']
		else:
			order[::2] = self.data['ln(PI)_minima_idx']
			order[1::2] = self.data['ln(PI)_maxima_idx']
		assert(np.all([order[i] <= order[i+1] for i in xrange(len(order)-1)])), 'Local maxima and minima not sorted correctly, try adjusting the value of smooth'

	def thermo(self, bool props=True, bool complete=False):
		"""
		Integrate the lnPI distribution, etc. and compute average thermodynamic properties of each phase identified.

		This adds F.E./kT, nn_mom, un_mom, n1, n2, ..., x1, x2, ..., u, and density keys to data['thermo'][phase_idx] for each phase. Does not check "safety" of the calculation, use is_safe() for that.
		Free energy is computed as -ln(sum(PI[N1]/PI[N1=0])).

		Parameters
		----------
		props : bool
			If True then computes the extensive properties, otherwise just integrates lnPI (free energy) for each phase (default=True)
		complete : bool
			If True then compute properties of entire distribution, ignoring phase segmentation of lnPI surface (default=False)

		"""

		cdef int p, i, j, k, m, q, left, right, min_ctr = 0, nphases = 0
		cdef double lnX, sum_prob

		if (not complete):
			try:
				self.normalize()
				self.relextrema()
			except Exception as e:
				raise Exception ('Unable to find relative extrema and normalize ln(PI) : '+str(e))
			nphases = len(self.data['ln(PI)_maxima_idx'])
		else:
			self.normalize()
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


	def temp_mu_extrap(self, double target_beta, np.ndarray[np.double_t, ndim=1] target_mus, int order=1, double cutoff=10.0, override=False, clone=True, skip_mom=False):
		"""
		Use temperature and chemical potential extrapolation to estimate lnPI and other extensive properties from current conditions (mu_1, beta).
		Should do reweighting (if desired) first, then call this function to extrapolate to other temperatures and mu_2, ..., mu_N.

		Parameters
		----------
		target_beta : double
			1/kT of the desired distribution
		target_mus : ndarray
			Desired chemical potentials of species 2 through N
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

		assert (len(target_mus) == self.data['nspec']-1), 'Must specify mu values for all components 2-N'

		orig_mu = copy.copy(self.metadata['mu_ref'][1:])
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
				tmp_hist._temp_dmu_extrap_2(target_beta, target_dmu, cutoff, override, skip_mom)
			except Exception as e:
				raise Exception('Unable to extrapolate : '+str(e))
		else:
			raise Exception('No implementation for temperature + dMu extrapolation of order '+str(order))

		tmp_hist.data['curr_beta'] = target_beta
		tmp_hist.data['curr_mu'][1:] = tmp_hist.data['curr_mu'][0] + target_dmu

		# Renormalize afterwards as well
		tmp_hist.normalize()

		return tmp_hist


	# def find_phase_eq(self, double lnZ_tol, double mu_guess, double beta=0.0, object mu=[], int extrap_order=1, double cutoff=10.0, bool override=False):
	# def temp_mu_extrap_multi(self, np.ndarray[np.double_t, ndim=1] target_betas, np.ndarray[np.double_t, ndim=2] target_mus, int order=1, double cutoff=10.0, override=False, skip_mom=False):

histogram._cy_normalize = types.MethodType(_cython_normalize, None, histogram)
histogram._cy_reweight = types.MethodType(_cython_reweight, None, histogram)

if __name__ == '__main__':
	print "gc_hist.pyx"

	"""

	* Tutorial:

	To compute the thermodynamic properties of a distribution at the current temperature

	1. Instantiate histogram object from file
	2. Reweight to desired chemical potential (mu1)
	3. Use thermo() to get thermodynamic properties of each phase
	4. Call is_safe() to check that the ln(PI) distribution extends far enough to trust this result

	To compute to another temperature/mu

	1. Instantiate histogram object from file
	2. Reweight to desired chemical potential (mu1)
	3. Call histogram = temp_extrap() with the appropriate flag set to either modify self or create a copy / use temp_mu_extrap() if extrapolating in temperature and mu_2, mu_3, ...
	4. histogram.thermo()
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
