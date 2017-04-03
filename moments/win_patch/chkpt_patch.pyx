"""@docstring
@brief Tools for patching windows (Nmin < N < Nmax) together to build a single flat histogram from FHMCSimulation simulations using their checkpoints
@author Nathan A. Mahynski
@date 03/15/2017
@filename checkpt_patch.pyx
"""

import operator, sys, re, os, cython, types, copy, time, json
import numpy as np

cimport numpy as np
cimport cython

from scipy.optimize import fmin
from netCDF4 import Dataset
from os import listdir
from os.path import isfile, join
from numpy import ndarray
from numpy cimport ndarray
from cpython cimport bool

cdef inline double double_max(double a, double b): return a if a > b else b
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport fabs

np.seterr(divide='raise', over='raise', invalid='raise') # Any sort of problem (except underflow), raise an exception

def tryint(s):
	"""
	Test if something is an integer or not.  If so, return its integer form, otherwise itself

	Parameters
	----------
	s : str
		Thing to test if integer or not

	Returns
	int
		Integer form if possible, else string form

	"""

	try:
		return int(s)
	except:
		return s

@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double spec_exp (double a, double b):
	"""
	Compute the natural logarithm of the sum of a pair of exponentials.  i.e. ln(exp(a) + exp(b))

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
cdef _cython_normalize_lnPI (self):
	"""
	Cythonized normalization of window's lnPI.

	"""

	cdef double lnNormPI = -sys.float_info.max
	cdef int i
	for i in xrange(len(self.lnPI)):
		lnNormPI = spec_exp (lnNormPI, self.lnPI[i])
	self.lnPI = self.lnPI - lnNormPI

class local_hist (object):
	def __init__ (self, fname):
		"""
		Initialize a local histogram of information stored during simulation at each N.
		Normalizes the rows because it is assumed this is loaded from a checkpoint, which does not contain normalized data.

		Parameters
		----------
		fname : str
			Filename to load histogram from

		"""

		try:
			self.load(fname)
			self.normalize()
		except Exception as e:
			raise Exception ('Unable to load local histogram from '+fname+' : '+str(e))

	def clear (self):
		"""
		Clear data

		"""

		self.ub = np.array([])
		self.lb = np.array([])
		self.bw = np.array([])
		self.h = []
		self.win_start = 0
		self.win_end = 0

	def load (self, fname):
		"""
		Load information from filename (.dat)

		"""

		self.clear()

		# Metadata first
		with open(fname, 'r') as f:
			for line in f:
				if (line[0] == "#"):
					if ("species_total_upper_bound" in line or "species_1_upper_bound" in line):
						info = line.strip().split(":")
						self.win_end = int(info[len(info)-1])
					elif ("species_total_lower_bound" in line or "species_1_lower_bound" in line):
						info = line.strip().split(":")
						self.win_start = int(info[len(info)-1])
				else:
					break

		assert (self.win_start < self.win_end), 'Bounds out of order'

		# Raw data next
		with open(fname, 'r') as f:
			next = None
			for line in f:
				if (line[0] == "#" and next != 'h'):
					if ("Bin widths for each" in line):
						next = 'bw'
					elif ("Bin lower bound for each" in line):
						next = 'lb'
					elif ("Bin upper bound for each" in line):
						next = 'ub'
					elif ("Unnormalized histogram for each" in line):
						next = 'h'
					else:
						next = None
				else:
					if (next == 'bw'):
						self.bw = np.array([float(x) for x in line.split('\t') if x != '\n'])
					elif (next == 'lb'):
						self.lb = np.array([float(x) for x in line.split('\t') if x != '\n'])
					elif (next == 'ub'):
						self.ub = np.array([float(x) for x in line.split('\t') if x != '\n'])
					elif (next == 'h'):
						self.h.append(np.array([float(x) for x in line.split('\t') if x != '\n']))
					else:
						pass

		assert (len(self.lb) == len(self.ub)), 'Bad bounds in local_hist'
		assert (len(self.lb) == len(self.bw)), 'Bad bin width in local_hist'

	def merge (self, other, other_weight, skip_hist=False):
		"""
		Merge two histograms.

		Ensures/requires alignment of histograms when overlapped. Does not renormalize.

		Parameters
		----------
		other : local_hist
			Other local_hist object to combine with
		other_weight : double
			0 <= w <= 1, relative to these values
		skip_hist : bool
			Whether or not to skip merging histograms, if skipping, fills all values with 1's (default=False)

		"""

		assert (other_weight >= 0 and other_weight <= 1), 'Weight out of range'
		# Start from the lowest lb and go to the highest ub
		new_start = np.min([self.win_start, other.win_start])
		new_end = np.max([self.win_end, other.win_end])
		new_bw = np.zeros(new_end-new_start+1, dtype=np.float64)
		new_lb = np.zeros(new_end-new_start+1, dtype=np.float64)
		new_ub = np.zeros(new_end-new_start+1, dtype=np.float64)
		new_h = []

		for n in xrange(new_start, new_end+1):
			belong_self = False
			belong_other = False
			if (n >= self.win_start and n <= self.win_end):
				belong_self = True
			if (n >= other.win_start and n <= other.win_end):
				belong_other = True

			if (belong_self and not belong_other):
				# Just take distribution from self
				new_bw[n-new_start] = self.bw[n-self.win_start]
				new_lb[n-new_start] = self.lb[n-self.win_start]
				new_ub[n-new_start] = self.ub[n-self.win_start]
				new_h.append(self.h[n-self.win_start])
				if (skip_hist):
					new_h[-1].fill(1)
			elif (belong_other and not belong_self):
				# Just take distribution from other
				new_bw[n-new_start] = other.bw[n-other.win_start]
				new_lb[n-new_start] = other.lb[n-other.win_start]
				new_ub[n-new_start] = other.ub[n-other.win_start]
				new_h.append(other.h[n-other.win_start])
				if (skip_hist):
					new_h[-1].fill(1)
			elif (belong_self and belong_other):
				if (skip_hist):
					new_bw[n-new_start] = self.bw[n-self.win_start]
					new_lb[n-new_start] = np.min([self.lb[n-self.win_start], other.lb[n-other.win_start]])
					new_ub[n-new_start] = np.max([self.ub[n-self.win_start], other.ub[n-other.win_start]])
					tot_bins = int(np.ceil((new_ub[n-new_start] - new_lb[n-new_start])/new_bw[n-new_start]))
					if (fabs(((new_ub[n-new_start] - new_lb[n-new_start])/new_bw[n-new_start]) - tot_bins) < 1.0e-8):
						tot_bins += 1 # Include endpoint
					new_h.append(np.zeros(tot_bins, dtype=np.float64))
					new_h[-1].fill(1)
				else:
					assert (fabs(self.bw[n-self.win_start] - other.bw[n-other.win_start]) < 1.0e-8), 'local_hist objects have different bin widths'
					x = fabs((self.lb[n-self.win_start] - other.lb[n-other.win_start])/self.bw[n-self.win_start])
					assert (fabs(x - np.round(x)) < 1.0e-8), 'Bin alignment error'
					x = fabs((self.ub[n-self.win_start] - other.ub[n-other.win_start])/self.bw[n-self.win_start])
					assert (fabs(x - np.round(x)) < 1.0e-8), 'Bin alignment error'
					new_bw[n-new_start] = self.bw[n-self.win_start]
					new_lb[n-new_start] = np.min([self.lb[n-self.win_start], other.lb[n-other.win_start]])
					new_ub[n-new_start] = np.max([self.ub[n-self.win_start], other.ub[n-other.win_start]])

					tot_bins = int(np.ceil((new_ub[n-new_start] - new_lb[n-new_start])/new_bw[n-new_start]))
					if (fabs(((new_ub[n-new_start] - new_lb[n-new_start])/new_bw[n-new_start]) - tot_bins) < 1.0e-8):
						tot_bins += 1 # Include endpoint
					new_h.append(np.zeros(tot_bins, dtype=np.float64))

					for i in xrange(tot_bins):
						x = i*new_bw[n-new_start]+new_lb[n-new_start]
						if (x >= self.lb[n-self.win_start] and x <= self.ub[n-self.win_start]):
							bin = int(np.ceil((x-self.lb[n-self.win_start])/self.bw[n-self.win_start]))
							assert (bin <= len(self.h[n-self.win_start]) and bin >= 0), 'Bin calculation error'
							if (bin == len(self.h[n-self.win_start])):
								bin -= 1 # Rounding at max bin
							a = self.h[n-self.win_start][bin]
						else:
							a = 0.0

						if (x >= other.lb[n-other.win_start] and x <= other.ub[n-other.win_start]):
							bin = int(np.ceil((x-other.lb[n-other.win_start])/other.bw[n-other.win_start]))
							assert (bin <= len(other.h[n-other.win_start]) and bin >= 0), 'Bin calculation error'
							if (bin == len(other.h[n-other.win_start])):
								bin -= 1 # Rounding at max bin
							b = other.h[n-other.win_start][bin]
						else:
							b = 0.0

						new_h[n-new_start][i] = a*(1.0-other_weight) + b*(other_weight)
			else:
				raise Exception ('Bounds error in merging local_hist objects')

		self.ub = copy.deepcopy(new_ub)
		self.lb = copy.deepcopy(new_lb)
		self.bw = copy.deepcopy(new_bw)
		self.h = copy.deepcopy(new_h)
		self.win_start = new_start
		self.win_end = new_end

	def normalize (self):
		"""
		Normalize the histogram.

		Though this should be superfluous if initial histograms are already normalized. If not, should normalized, then merge.

		"""

		for row in self.h:
			sum = np.sum(row)
			new_row = np.array(row)/sum
			row = new_row.tolist()

class window (object):
	"""
	Class to store histogram information from a window of WL-TMMC simulations.

	Also contains methods to perform basic manipulations.

	"""

	def __init__ (self, lnPI_fname, mom_fname, ehist_fname, pkhist_prefix, offset=2, smooth=False):
		"""
		Instatiate the class

		Parameters
		----------
		lnPI_fname : str
			File (.dat) containing lnPI distribution from WL-TMMC simulations
		mom_fname : str
			File (.dat) containing N_i^jN_k^mU^p moments
		ehist_fname : str
			Energy histogram filename (.dat) containing the histogram of energy at each N
		pkhist_prefix : str
			Particle histogram filename (.dat) prefix containing the histogram of particle numbers at each N
		offset : int
			The amount to trim off the edge of window overlap when patching (default=2)
		smooth : bool
			Whether or not to smooth the data between this histogram and another that is merged into it. No smoothing just uses histogram at lower N's values. (default=False)

		"""

		self.clear()

		self.lnPI_fname = lnPI_fname
		self.mom_fname = mom_fname
		self.ehist_fname = ehist_fname
		self.pkhist_prefix = pkhist_prefix
		self.offset = offset
		self.smooth = smooth

		assert (self.lnPI_fname[len(lnPI_fname)-4:] == '.dat'), 'Expects .dat file'
		assert (self.mom_fname[len(mom_fname)-4:] == '.dat'), 'Expects .dat file'
		assert (self.ehist_fname[len(ehist_fname)-4:] == '.dat'), 'Expects .dat file'
		assert (self.offset >= 1), 'Offset must be >= 1'

		self.reload()

	def __repr__ (self):
		"""
		Represent self

		Returns
		-------
		str
			"lnPI_fname::mom_fname::ehist_fname::pkhist_prefix-[lowerNBound, upperNBound]"

		"""

		return self.lnPI_fname+"::"+self.mom_fname+"::"+self.ehist_fname+"::"+self.pkhist_prefix+"-["+str(self.lb)+","+str(self.ub)+"]"

	def __lt__ (self, other):
		"""
		Sort these objects based on their lower N bounds

		Parameters
		----------
		other : window
			window object to combine with

		"""

		return self.lb < other.lb

	def clear (self):
		"""
		Clear all data in the class

		"""

		self.lnPI = np.array([])
		self.max_order = 0
		self.mom = np.array([])
		self.pk_hist = []
		self.e_hist = []
		self.lb = 0
		self.ub = 0
		self.nspec = 0
		self.V = 0
		self.op_name = ""

	def normalize (self):
		"""
		Normalize so that lnPI represents a normalized set of PI values

		"""

		self._cy_normalize ()

	def reload (self):
		"""
		Reload data from .dat files.
		This also looks at the raw data to figure out what the order parameter was.

		"""

		self.clear()

		# Get metadata from moments file
		with open(self.mom_fname, 'r') as f:
			for line in f:
				if (line[0] == "#"):
					if ("species_total_upper_bound" in line):
						if (self.op_name == "" or self.op_name == "N_{tot}"):
							self.op_name = "N_{tot}"
						else:
							raise Exception ("Order parameter seems to change inside a window")
						info = line.strip().split(":")
						self.ub = int(info[len(info)-1])
					elif ("species_1_upper_bound" in line):
						if (self.op_name == "" or self.op_name == "N_{1}"):
							self.op_name = "N_{1}"
						else:
							raise Exception ("Order parameter seems to change inside a window")
						info = line.strip().split(":")
						self.ub = int(info[len(info)-1])
					elif ("species_total_lower_bound" in line):
						if (self.op_name == "" or self.op_name == "N_{tot}"):
							self.op_name = "N_{tot}"
						else:
							raise Exception ("Order parameter seems to change inside a window")
						info = line.strip().split(":")
						self.lb = int(info[len(info)-1])
					elif ("species_1_lower_bound" in line):
						if (self.op_name == "" or self.op_name == "N_{1}"):
							self.op_name = "N_{1}"
						else:
							raise Exception ("Order parameter seems to change inside a window")
						info = line.strip().split(":")
						self.lb = int(info[len(info)-1])
					elif ("volume" in line):
						info = line.strip().split(":")
						self.V = float(info[len(info)-1])
					elif ("max_order" in line):
						info = line.strip().split(":")
						self.max_order = int(info[len(info)-1])
					elif ("number_of_species" in line):
						info = line.strip().split(":")
						self.nspec = int(info[len(info)-1])
				else:
					break

		# Load information
		self.lnPI = np.loadtxt(self.lnPI_fname, dtype=np.float, comments="#", unpack=True)
		self.mom = np.loadtxt(self.mom_fname, dtype=np.float, comments="#", unpack=True)
		self.mom = self.mom[1:]/self.mom[1] # Trim order parameter column, and normalize
		assert (self.mom.shape[1] == len(self.lnPI)), 'Inconsistent number of entries in files'
		self.e_hist = local_hist (self.ehist_fname) # This is normalized internally when instantiated
		self.pk_hist = []
		for i in xrange(self.nspec):
			self.pk_hist.append(local_hist (self.pkhist_prefix+'_'+str(i+1)+'.dat'))  # This is normalized internally when instantiated

	def merge (self, other, skip_hist=False):
		"""
		Merge this window with another and store in this object (self is modified).

		Automatically renormalizes the histograms of particle number and energy, just in case.

		Parameters
		----------
		other : window
			window object to combine with - this should be a lower range of N than this one
		skip_hist : bool
			Whether or not to skip merging histograms, if skipping, fills all values with 1's (default=False)

		"""

		assert (self.nspec == other.nspec), 'Number of components different, cannot merge'
		shift, err2 = patch_window_pair (self, other)
		self.lnPI += shift

		assert (self.lb > other.lb), 'Can only patch from high '+self.op_name+' to lower'
		assert (self.offset == other.offset), 'Cannot patch, inconsistent offsets'
		assert (self.offset >= 1), 'Invalid offset found during merge'
		cdef int index = other.ub - self.lb + 1, i
		self.lb = other.lb

		if (self.smooth):
			# Smooth the data
			partA = other.lnPI[:len(other.lnPI)-index+self.offset]
			o_B = other.lnPI[len(other.lnPI)-index+self.offset:len(other.lnPI)-other.offset]
			s_B = self.lnPI[self.offset:index-other.offset]
			o_W = np.arange(len(o_B), 0, -1, dtype=np.float64) # Weight based on position
			s_W = np.arange(1, len(s_B)+1, dtype=np.float64) # Weight based on position
			partB = (o_B*o_W + s_B*s_W)/(o_W+s_W)
			partC = self.lnPI[index-other.offset:]
			self.lnPI = np.concatenate([partA, partB, partC])

			partA = other.mom[:,:other.mom.shape[1]-index+self.offset]
			o_B = other.mom[:,other.mom.shape[1]-index+self.offset:other.mom.shape[1]-other.offset]
			s_B = self.mom[:,self.offset:index-other.offset]
			o_Wt = np.arange(o_B.shape[1], 0, -1, dtype=np.float64) # Weight based on position
			o_W = copy.copy(o_Wt)
			o_W = o_W.reshape((1,o_B.shape[1]))
			for i in xrange(other.mom.shape[0]-1):
				o_W = np.vstack([o_W, o_Wt])
			s_Wt = np.arange(s_B.shape[1], 0, -1, dtype=np.float64) # Weight based on position
			s_W = copy.copy(s_Wt)
			s_W = s_W.reshape((1,s_B.shape[1]))
			for i in xrange(self.mom.shape[0]-1):
				s_W = np.vstack([s_W, s_Wt])
			partB = (o_B*o_W + s_B*s_W)/(o_W+s_W)
			partC = self.mom[:,index-other.offset:]
			self.mom = np.hstack([partA, partB, partC])

			self.e_hist.merge(other.e_hist, 0.5, skip_hist) # Just merge 'evenly' rather by position-dependent weight
			self.e_hist.normalize()
			for i in xrange(self.nspec):
				self.pk_hist[i].merge(other.pk_hist[i], 0.5, skip_hist)
				self.pk_hist[i].normalize()
		else:
			# Simply concatenate the data
			self.lnPI = np.concatenate([other.lnPI[:len(other.lnPI)-other.offset], self.lnPI[index-self.offset:]])
			self.mom = np.hstack([other.mom[:,:other.mom.shape[1]-other.offset], self.mom[:,index-self.offset:]])
			self.e_hist.merge(other.e_hist, 1.0, skip_hist) # Use lower N results completely when overlap occurs
			self.e_hist.normalize()
			for i in xrange(self.nspec):
				self.pk_hist[i].merge(other.pk_hist[i], 1.0, skip_hist)
				self.pk_hist[i].normalize()

		return shift, err2

	def to_nc (self, fname):
		"""
		Print the composite information to a netCDF4 file

		Parameters
		----------
		fname : str
			Name of file to print to (should contain .nc suffix)

		"""

		dataset = Dataset(fname, "w", format="NETCDF4")
		dataset.history = 'Created ' + time.ctime(time.time())
		dataset.volume = self.V
		dataset.nspec = self.nspec
		dataset.max_order = self.max_order

		dim_OP = dataset.createDimension(self.op_name, len(self.lnPI))
		dim_i = dataset.createDimension("i", self.nspec)
		dim_j = dataset.createDimension("j", self.max_order+1)
		dim_k = dataset.createDimension("k", self.nspec)
		dim_m = dataset.createDimension("m", self.max_order+1)
		dim_p = dataset.createDimension("p", self.max_order+1)

		var_OP = dataset.createVariable(self.op_name, np.int, (self.op_name,))
		var_lnPI = dataset.createVariable("ln(PI)", np.float64, (self.op_name,))
		var_i = dataset.createVariable("i", np.int, ("i",))
		var_j = dataset.createVariable("j", np.int, ("j",))
		var_k = dataset.createVariable("k", np.int, ("k",))
		var_m = dataset.createVariable("m", np.int, ("m",))
		var_p = dataset.createVariable("p", np.int, ("p",))
		var_moments = dataset.createVariable("N_{i}^{j}*N_{k}^{m}*U^{p}", np.float64, ("i", "j", "k", "m", "p", self.op_name,))

		var_OP[:] = np.arange(self.lb, self.ub+1)
		var_i[:] = np.arange(1, self.nspec+1)
		var_j[:] = np.arange(0, self.max_order+1)
		var_k[:] = np.arange(1, self.nspec+1)
		var_m[:] = np.arange(0, self.max_order+1)
		var_p[:] = np.arange(0, self.max_order+1)
		var_lnPI[:] = self.lnPI

		cdef int address = 0, i, j, k, m, p
		for i in xrange(self.nspec):
			for j in xrange(self.max_order+1):
				for k in xrange(self.nspec):
					for m in xrange(self.max_order+1):
						for p in xrange(self.max_order+1):
							var_moments[i,j,k,m,p,:] = self.mom[address]
							address += 1

		# Histograms for pk number and energy
		max_bin = 0
		for n in xrange(len(dim_OP)):
			max_bin = np.max([max_bin, len(self.e_hist.h[n])])
			for i in xrange(self.nspec):
				max_bin = np.max([max_bin, len(self.pk_hist[i].h[n])])
		dim_bin = dataset.createDimension("bin", max_bin)

		var_pkhist = dataset.createVariable("P_{N_i}("+self.op_name+")", np.float64, ("i", self.op_name, "bin",))
		var_pkhist_lb = dataset.createVariable("P_{N_i}("+self.op_name+")_{lb}", np.float64, ("i", self.op_name,))
		var_pkhist_ub = dataset.createVariable("P_{N_i}("+self.op_name+")_{ub}", np.float64, ("i", self.op_name,))
		var_pkhist_bw = dataset.createVariable("P_{N_i}("+self.op_name+")_{bw}", np.float64, ("i", self.op_name,))

		for i in xrange(self.nspec):
			var_pkhist_lb[i,:] = self.pk_hist[i].lb
			var_pkhist_ub[i,:] = self.pk_hist[i].ub
			var_pkhist_bw[i,:] = self.pk_hist[i].bw
			for n in xrange(len(dim_OP)):
				var_pkhist[i,n,0:len(self.pk_hist[i].h[n])] = self.pk_hist[i].h[n]
				var_pkhist[i,n,len(self.pk_hist[i].h[n]):] = 0.0

		var_ehist = dataset.createVariable("P_{U}("+self.op_name+")", np.float64, (self.op_name, "bin",))
		var_ehist_lb = dataset.createVariable("P_{U}("+self.op_name+")_{lb}", np.float64, (self.op_name,))
		var_ehist_ub = dataset.createVariable("P_{U}("+self.op_name+")_{ub}", np.float64, (self.op_name,))
		var_ehist_bw = dataset.createVariable("P_{U}("+self.op_name+")_{bw}", np.float64, (self.op_name,))

		var_ehist_lb[:] = self.e_hist.lb
		var_ehist_ub[:] = self.e_hist.ub
		var_ehist_bw[:] = self.e_hist.bw
		for n in xrange(len(dim_OP)):
			var_ehist[n,0:len(self.e_hist.h[n])] = self.e_hist.h[n]
			var_ehist[n,len(self.e_hist.h[n]):] = 0.0

		dataset.close()

window._cy_normalize = types.MethodType(_cython_normalize_lnPI, None, window)

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double window_patch_error (double x, np.ndarray[np.float_t, ndim=1] this_lnPI, np.ndarray[np.float_t, ndim=1] other_lnPI):
	"""
	Estimate the square error from two overlapping lnPI distributions

	Parameters
	----------
	x : double
		Shift to apply to selfData to try to match otherData (= selfData + x)
	this_lnPI : ndarray
		lnPI to be shifted by x to match otherData
	other_lnPI : ndarray
		lnPI distribution trying to match

	Returns
	-------
	double
		Total square error

	"""

	cdef double e2 = 0.0
	cdef int i
	for i in xrange(len(this_lnPI)):
		e2 += ((this_lnPI[i]+x)-other_lnPI[i])**2
	return e2

@cython.boundscheck(False)
@cython.cdivision(True)
cdef patch_window_pair (window_hist1, window_hist2, double ftol=0.000001):
	"""
	Patch to another histogram assuming same boltzmann factors.

	Parameters
	----------
	window_hist1 : window
		window_hist object to combine with window_hist2. Should be greater than window_hist2.
	window_hist2 : window
		window_hist object to combine with window_hist1. Should be less than window_hist1.
	ftol : double
		Patching tolerance (default=0.000001)

	Returns
	-------
	double, double
		Shift necessary for window_hist1 to match window_hist2, error^2/number of overlapping points (less the offset)

	"""

	# Double check overall bounds
	assert (window_hist1.lb > window_hist2.lb), 'Histograms out of order, cannot patch' # Guarantee data1 > data2
	assert (window_hist1.ub > window_hist2.ub), 'Histograms out of order, cannot patch' # Guarantee data1 > data2
	assert (window_hist1.lb < window_hist2.ub), 'Histograms do not overlap, cannot patch' # Guarantee to overlap

	# Align the overlapping data
	cdef int index = window_hist2.ub - window_hist1.lb + 1
	data_slice1 = window_hist1.lnPI[window_hist1.offset:index-window_hist1.offset] # Exclude the beginning and end points from fitting because of artificial edge effects there
	data_slice2 = window_hist2.lnPI[len(window_hist2.lnPI)-index+window_hist1.offset:len(window_hist2.lnPI)-window_hist1.offset] # Exclude the end points from fitting because of artificial edge effects there

	assert (len(data_slice1) > 1), 'Error, unable to patch windown because there is no overlap'
	assert (len(data_slice2) > 1), 'Error, unable to patch windows because there is no overlap'

	cdef double lnPIshift_guess = data_slice2[0] - data_slice1[0]

	# Optimize
	full_out = fmin(window_patch_error, lnPIshift_guess, ftol=ftol, args=(data_slice1, data_slice2, ), maxiter=10000, maxfun=10000, full_output=True)

	if (full_out[:][4] != 0):
		raise Exception("Error, unable to mimize "+str(window_hist1)+" and "+str(window_hist2)+" : "+str(full_out))

	return full_out[0][0], full_out[1]/len(data_slice1)

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef patch_all_windows (fnames, out_fname="composite.nc", log_fname="patch.log", offset=2, smooth=False, tol=np.inf, skip_hist=False, last_safe_idx=-1):
	"""
	Take a series of filenames for files containing the raw histogram data for different windows
	and iteratively patch neighbors.

	The result is a sequence of shifts in lnPI for each window and the single, self-consistent, normalized histogram is printed to file. Automatically sorts bounds.

	Parameters
	----------
	fnames : list
		List of (lnPI_fname, moment_fname, ehist_fname, pkhist_prefix) filenames to create window_hist's
	out_fname : str
		Name of final composite histogram (default=composite.nc)
	log_fname : str
		Filename to store the shifts of each lnPI histogram in (default=patch.log)
	offset : double
		The amount to trim off the edge of window overlap when patching (default=2)
	smooth : bool
		Smooth the overlapping lnPI and N1, etc.? (default=False)
	tol : double
		Max error tolerance for mean err^2 in ln(PI) patching. (default=np.inf)
	skip_hist : bool
		Whether to disregard pk and U histograms (default=False)
	last_safe_idx : int
		Index of sorted histogram that is safe to patch into (i.e. below threshold).  By default this is -1 (test all), but routine will recursively call itself to locate this properly.  Do not specify yourself.

	Returns
	-------
	str, double, double
		Name of the histogram responsible for the worst patching error, Value of the worst normalized, squared patching error

	"""

	cdef int i, next, end = 0, start

	histograms = []
	for name_l, name_mom, name_e, name_p in fnames:
		try:
			histograms.append(window(name_l, name_mom, name_e, name_p, offset, smooth))
		except Exception as e:
			raise Exception ('Unable to generate patch sequence : '+str(e))

	if (last_safe_idx < 0):
		end = len(histograms)-1
	else:
		end = last_safe_idx

	# Sort based on lower bound, and verify that no more than 2 histograms overlap at once
	histograms.sort()
	for i in xrange(0, end):
		if (i < len(histograms)-2):
			# Ensure its upper bound overlaps the lower bound of the next
			if (histograms[i].ub <= histograms[i+1].lb):
				raise Exception ("Histograms from "+str(histograms[i])+" and "+str(histograms[i+1])+" do not overlap")
			# But does not overlap the next one
			if (histograms[i].ub > histograms[i+2].lb):
				raise Exception ("Histograms from "+str(histograms[i])+", "+str(histograms[i+1])+", and "+str(histograms[i+2])+" overlap")
		else:
			# Ensure its upper bound overlaps the lower bound of the next
			if (histograms[i].ub <= histograms[i+1].lb):
				raise Exception ("Histograms from "+str(histograms[i])+" and "+str(histograms[i+1])+" do not overlap")

	# Shift, and combine iteratively
	f = open(log_fname, 'w')
	next = end-1
	err_vals = {}
	while (next >= 0):
		lnPIshift, norm_err2 = histograms[end].merge(histograms[next], skip_hist)
		err_vals[str(histograms[next])] = norm_err2
		f.write("Patching {"+str(histograms[next])+"} into {"+str(histograms[end])+"} : "+str(lnPIshift)+"\n")
		next -= 1

	# Find first location of error exceeding tolerance, and call self again with this bound
	for i in xrange(end):
		if (err_vals[str(histograms[i])] > tol):
			#f.write('ln(PI) error tolerance exceeded for '+str(histograms[i])+', repatching below this: '+str(err_vals[str(histograms[i])])+' > '+str(tol)+'\n')
			patch_all_windows(fnames, out_fname, log_fname, offset, smooth, tol, skip_hist, i)
	f.close()

	# Normalize overall histogram
	if (len(histograms) == 1):
		max_err = [str(histograms[0]), 0.0]
	else:
		max_err = max(err_vals.iteritems(), key=operator.itemgetter(1))
	histograms[end].normalize()

	# Double check
	cdef double isum = -sys.float_info.max
	for i in xrange(len(histograms[end].lnPI)):
		isum = spec_exp (isum, histograms[end].lnPI[i])
	isum = exp(isum)

	if (np.fabs(isum - 1.0) > 2.0e-12):
		raise Exception("Failed to patch: composite PI sums to "+str(sum)+" which differs from 1 by "+str(np.fabs(sum - 1.0)))

	# Print composite
	histograms[end].to_nc(out_fname)

	# Return max patching err and name of histogram responsible for it
	return max_err[0], max_err[1]

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef get_patch_sequence (idir):
	"""
	Look through local windows (numbered) to find the histograms to patch, based on their checkpoints (/checkpt directory)

	Returns ordered, continuous list of tuple containing the filenames for each window class.

	Parameters
	----------
	idir : str
		Directory to look for windows in (directories: 1, 2, 3...)

	Returns
	-------
	list
		Ordered list of filenames: (lnPI_fname, mom_fname, ehist_fname, pkhist_prefix)

	"""

	# Trim trailing '/'
	if (idir[len(idir)-1] == '/'):
		dir = idir[:len(idir)-1]
	else:
		dir = copy.copy(idir)

	# Find the directories for each window
	oD = sorted([ tryint(f) for f in listdir(dir) if not isfile(join(dir,f)) ])
	only_dirs = [dir+'/'+str(d)+'/checkpt' for d in oD if "checkpt" in os.listdir(dir+'/'+str(d))]

	lnPI_fname = []
	mom_fname = []
	ehist_fname = []
	pkhist_prefix = []

	for d in only_dirs:
		files = listdir(d)
		try:
			fname = open(d+'/state.json', 'r')
			data = json.load(fname)
			fname.close()
		except:
			raise Exception ('Checkpoint status file could not be located in : '+str(d))
		else:
			# Check if TMMC stage has started
			if data["crossoverDone"] is True:
				found = {'tmmc':False, 'mom':False, 'eh':False, 'ph':False, 'measured':False}
				fn = {'tmmc':'', 'mom':'', 'eh':'', 'ph':''}
				for f in files:
					if ("tmmc_lnPI.dat" in f):
						found['tmmc'] = True
						fn['tmmc'] = d+"/"+f
					if ("extMom.dat" in f):
						found['mom'] = True
						fn['mom'] = d+"/"+f
						f = open(fn['mom'], 'r') # Also check that all entries were measured
						dummy = np.loadtxt(f, usecols=(1,), unpack=True) # Entries are order_param, then N1^0*...N_n^0*^0 which is just the counter for how many times it was measured
						f.close()
						if (np.all(dummy >= 1.0)):
							found['measured'] = True
					if ("eHist.dat" in f):
						found['eh'] = True
						fn['eh'] = d+"/"+f
					if ("pkHist_1.dat" in f): # Only bother to look for species 1
						found['ph'] = True
						fn['ph'] = d+"/pkHist"
				if (np.all([found[i] for i in found])):
					lnPI_fname.append(fn['tmmc'])
					mom_fname.append(fn['mom'])
					ehist_fname.append(fn['eh'])
					pkhist_prefix.append(fn['ph'])
				else:
					break # Do not continue after first failure in order to avoid getting windows out of order

	return zip(lnPI_fname, mom_fname, ehist_fname, pkhist_prefix)

if __name__ == '__main__':
	print "checkpt_patch.pyx"

	"""

	* Tutorial:

	1. Run get_patch_sequence() in simulation directory where each window is stored in a
		local directory labelled numerically (1, 2, 3,...)
	2. Feed the output to patch_all_windows() to patch together all results

	* Notes:

	If desired, an intermediate check can be applied between steps 1 and 2 to remove
		windows deemed "unequilibrated".  This is designed to operate with FHMCSimulation code
		by N. A. Mahynski.  The expected file storage format reflect this, but can be
		modified as necessary. I would recommend making a new module for other interfaces.

	"""
