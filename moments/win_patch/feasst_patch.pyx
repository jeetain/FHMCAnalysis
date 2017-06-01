"""@docstring
@brief Tools for patching windows (Nmin < N < Nmax) together to build a single flat histogram from FEASST simulations
@author Nathan A. Mahynski
@date 05/23/2017
@filename feasst_patch.pyx
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

class window (object):
	"""
	Class to store histogram information from a window of WL-TMMC simulations.

	Also contains methods to perform basic manipulations.

	"""

	def __init__ (self, colMat_fname="colMat", extMom_fname="extMom_pr", offset=2, smooth=False):
		"""
		Instatiate the class

		Parameters
		----------
		colMat_fname : str
			File containing lnPI distribution from WL-TMMC simulations (default="colMat")
		extMom_fname : str
			File containing N_i^jN_k^mU^p moments (default="extMom_pr")
		offset : int
			The amount to trim off the edge of window overlap when patching (default=2).  If offset = 0, assumes that windows overlap by a single value of the order parameter at its edge.
		smooth : bool
			Whether or not to smooth the data between this histogram and another that is merged into it. No smoothing just uses histogram at lower N's values. (default=False)

		"""

		self.clear()

		self.colMat_fname = colMat_fname
		self.extMom_fname = extMom_fname
		self.offset = offset
		self.smooth = smooth

		assert (self.offset >= 0), 'Offset must be >= 0'

		self.reload()

	def __repr__ (self):
		"""
		Represent self

		Returns
		-------
		str
			"colMat_fname::extMom_fname-[lowerNBound, upperNBound]"

		"""

		return self.colMat_fname+"::"+self.extMom_fname+"-["+str(self.lb)+","+str(self.ub)+"]"

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
		self.mom_exp = np.array([])
		self.lb = 0
		self.ub = 0
		self.V = 0.0
		self.nspec = 0
		self.op_name = ""

	def normalize (self):
		"""
		Normalize so that lnPI represents a normalized set of PI values

		"""

		self._cy_normalize ()

	def reload (self):
		"""
		Reload data from files.
		This also looks at the raw data to figure out what the order parameter was.

		"""

		cdef long long unsigned int ctr, opIdx, nValues, i, j, k, m, p, num_moments
		cdef double Sum, SumSq

		self.clear()

		# Get metadata from moments file
		with open(self.extMom_fname, 'r') as f:
			for line in f:
				if (line[0] == "#"):
					if ("maxOrder" in line):
						#info = line.strip().replace(" ","").split(":")
						info = line.strip().split(" ")
						self.max_order = int(info[len(info)-1])
					elif ("nSpec" in line):
						#info = line.strip().replace(" ","").split(":")
						info = line.strip().split(" ")
						self.nspec = int(info[len(info)-1])
					elif ("orderParam" in line):
						#info = line.strip().replace(" ","").split(":")
						info = line.strip().split(" ")
						assert (str(info[len(info)-1]) == "nmol"), "FEASST requires total number of molecules as order parameter : "+str(info[len(info)-1])
						self.op_name = "N_{tot}" # this is the name used in rest of analysis, "nmol" is specific to FEASST
					elif ("volume" in line):
						#info = line.strip().replace(" ","").split(":")
						info = line.strip().split(" ")
						self.V = float(info[len(info)-1])
					elif ("nBin" in line):
						#info = line.strip().replace(" ","").split(":")
						info = line.strip().split(" ")
						nbins = int(info[len(info)-1])
					elif ("mMax" in line):
						#info = line.strip().replace(" ","").split(":")
						info = line.strip().split(" ")
						self.ub = int(np.floor(float(info[len(info)-1]))) # FEASST bin = 1 but reports at "midpoint"
					elif ("mMin" in line):
						#info = line.strip().replace(" ","").split(":")
						info = line.strip().split(" ")
						self.lb = int(np.ceil(float(info[len(info)-1]))) # FEASST bin = 1 but reports at "midpoint"
				else:
					break

		assert (self.ub-self.lb+1 == nbins), "Upper and lower bounds do not match number of bins in : "+str(self.extMom_fname)

		# Load information
		self.lnPI = np.loadtxt(self.colMat_fname, dtype=np.float, comments="#", unpack=True)[1] # second column in ln(PI) data
		self.mom = np.zeros((self.nspec*(self.max_order+1)*self.nspec*(self.max_order+1)*(self.max_order+1), nbins), dtype=np.float64)
		self.mom_exp = np.zeros((self.nspec*(self.max_order+1)*self.nspec*(self.max_order+1)*(self.max_order+1), 5), dtype=np.int32)

		dummy_mom = np.loadtxt(self.extMom_fname, dtype=np.float64, comments="#", unpack=False)
		ctr = 0
		num_moments = len(self.mom_exp)
		for row in dummy_mom:
			opIdx, nValues, Sum, SumSq, i, j, k, m, p = row
			momIdx = ctr%num_moments
			self.mom[momIdx, opIdx] = Sum/nValues
			self.mom_exp[momIdx] = [i,j,k,m,p]
			ctr += 1

		assert (self.mom.shape[1] == len(self.lnPI)), 'Inconsistent number of entries in files'

	def merge (self, other):
		"""
		Merge this window with another and store in this object (self is modified).

		Parameters
		----------
		other : window
			window object to combine with - this should be a lower range of N than this one

		"""

		assert (self.nspec == other.nspec), 'Number of components different, cannot merge'
		shift, err2 = patch_window_pair (self, other)
		self.lnPI += shift

		assert (self.max_order == other.max_order), 'Unequal maximum orders between windows, cannot merge'
		assert (self.V == other.V), 'Unequal volumes between windows, cannot merge'
		assert (self.op_name == other.op_name), 'Different order parameters between windows, cannot merge'
		assert (self.lb > other.lb), 'Can only patch from high '+self.op_name+' to lower'
		assert (self.offset == other.offset), 'Cannot patch, inconsistent offsets between windows'
		assert (self.offset >= 0), 'Invalid offset found during merge'
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
		else:
			# Simply concatenate the data
			self.lnPI = np.concatenate([other.lnPI[:len(other.lnPI)-other.offset], self.lnPI[index-self.offset:]])
			self.mom = np.hstack([other.mom[:,:other.mom.shape[1]-other.offset], self.mom[:,index-self.offset:]])

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

		cdef long long unsigned int address = 0, i, j, k, m, p, ii, jj, kk, mm, pp, nn
		for p in xrange(self.max_order+1):
			for m in xrange(self.max_order+1):
				for k in xrange(self.nspec):
					for j in xrange(self.max_order+1):
						for i in xrange(self.nspec):
							ii, jj, kk, mm, pp = self.mom_exp[address]
							# check that "unrolling" was correctly done
							if (not (i == ii and j == jj and k == kk and m == mm and p == pp)):
								raise Exception("Exponent indices do not match : "+str([i,j,k,m,p])+" vs "+str([ii,jj,kk,mm,pp]))
							var_moments[ii,jj,kk,mm,pp,:] = self.mom[address]
							address += 1

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
def patch_all_windows (fnames, **kwargs):
	"""
	Take a series of filenames for files containing the raw histogram data for different windows
	and iteratively patch neighbors.

	The result is a sequence of shifts in lnPI for each window and the single, self-consistent, normalized histogram is printed to file. Automatically sorts bounds.

	Parameters
	----------
	fnames : list
		List of (colMat_fname, extMom_fname) absolute paths to create window_hist's (see get_patch_sequence())
	Keyword Arguments :
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
		last_safe_idx : int
			Index of sorted histogram that is safe to patch into (i.e. below threshold).  By default this is -1 (test all), but routine will recursively call itself to locate this properly.  DO NOT specify yourself.

	Returns
	-------
	str, double, double
		Name of the histogram responsible for the worst patching error, Value of the worst normalized, squared patching error

	"""

	out_fname = kwargs.get('out_fname', "composite.nc")
	log_fname = kwargs.get('log_fname', "patch.log")
	offset = kwargs.get('offset', 2)
	smooth = kwargs.get('smooth', False)
	tol = kwargs.get('tol', np.inf)
	last_safe_idx = kwargs.get('last_safe_idx', -1)

	cdef int i, next, end = 0, start

	histograms = []
	for name_l, name_mom in fnames:
		try:
			histograms.append(window(colMat_fname=name_l, extMom_fname=name_mom, offset=offset, smooth=smooth))
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
		lnPIshift, norm_err2 = histograms[end].merge(histograms[next])
		err_vals[str(histograms[next])] = norm_err2
		f.write("Patching {"+str(histograms[next])+"} into {"+str(histograms[end])+"} : "+str(lnPIshift)+"\n")
		next -= 1

	# Find first location of error exceeding tolerance, and call self again with this bound
	for i in xrange(end):
		if (err_vals[str(histograms[i])] > tol):
			#f.write('ln(PI) error tolerance exceeded for '+str(histograms[i])+', repatching below this: '+str(err_vals[str(histograms[i])])+' > '+str(tol)+'\n')
			patch_all_windows (fnames, out_fname=out_fname, log_fname=log_fname, offset=offset, smooth=smooth, tol=tol, last_safe_idx=i)
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

	if (np.fabs(isum - 1.0) > 1.0e-10):
		raise Exception("Failed to patch: composite PI sums to "+str(isum)+" which differs from 1 by "+str(np.fabs(isum - 1.0)))

	# Print composite
	histograms[end].to_nc(out_fname)

	# Return max patching err and name of histogram responsible for it
	return max_err[0], max_err[1]

@cython.boundscheck(False)
@cython.cdivision(True)
def get_patch_sequence (idir, **kwargs):
	"""
	Look through local windows (numbered) to find the histograms to patch

	Returns ordered, continuous list of tuple containing the filenames for each window class.

	Parameters
	----------
	idir : str
		Directory to look for windows in (directories: 1, 2, 3...)
	Keyword Arguments :
		bound : int
			Last window to use (default=1000000)
		colMat_fname : str
			Name of file each window stored collection matrix / ln(PI) to (default="colMat")
		extMom_fname : str
			Name of file each window stored extensive moments to (default="extMom_pr")

	Returns
	-------
	list
		Ordered list of filenames: (colMat_fname, extMom_fname)

	"""

	cdef long unsigned int bound = kwargs.get('bound', 1000000)
	colMat_fname = kwargs.get("colMat_fname", "colMat")
	extMom_fname = kwargs.get("extMom_fname", "extMom_pr")

	# Trim trailing '/'
	if (idir[len(idir)-1] == '/'):
		dir = idir[:len(idir)-1]
	else:
		dir = copy.copy(idir)

	# Find the directories for each window
	oD = sorted([ tryint(f) for f in listdir(dir) if not isfile(join(dir,f)) ])
	only_dirs = [dir+'/'+str(d) for d in oD]

	lnPI_fname = []
	mom_fname = []

	for d in only_dirs:
		files = listdir(d)
		found = {'tmmc':False, 'mom':False,}
		fn = {'tmmc':'', 'mom':'',}
		for f in files:
			if (colMat_fname in f and '.bak' not in f):
				found['tmmc'] = True
				fn['tmmc'] = d+"/"+f
			if (extMom_fname in f and '.bak' not in f):
				found['mom'] = True
				fn['mom'] = d+"/"+f
		if (np.all([found[i] for i in found])):
			lnPI_fname.append(fn['tmmc'])
			mom_fname.append(fn['mom'])
		else:
			break # Do not continue after first failure in order to avoid getting windows out of order

	return zip(lnPI_fname, mom_fname)

@cython.boundscheck(False)
@cython.cdivision(True)
def get_patch_sequence_multicore (idir, **kwargs):
	"""
	Look through single directory to find the histograms to patch, named based on processor they ran on (numbered).

	Returns ordered, continuous list of tuple containing the filenames for each window class.

	Parameters
	----------
	idir : str
		Directory to look for windows in
	Keyword Arguments :
		bound : int
			Last window to use (default=1000000)
		colMat_pre : str
			Prefix of file each window stored collection matrix / ln(PI) to (default="colMat")
		colMat_suf : str
			Suffix of file each window stored collection matrix / ln(PI) to (default="")
		extMom_pre : str
			Prefix of file each window stored extensive moments to (default="extMom_pr_")
		extMom_suf : str
			Suffix of file each window stored extensive moments to (default="")
	Returns
	-------
	list
		Ordered list of filenames: (colMat_fname, extMom_fname)

	"""

	cdef long unsigned int bound = kwargs.get('bound', 1000000)
	colMat_pre = kwargs.get("colMat_pre", "colMat")
	colMat_suf = kwargs.get("colMat_suf", "")
	extMom_pre = kwargs.get("extMom_pre", "extMom_pr_")
	extMom_suf = kwargs.get("extMom_suf", "")

	# Trim trailing '/' if it exists to standardize
	if (idir[len(idir)-1] == '/'):
		dir = idir[:len(idir)-1]
	else:
		dir = copy.copy(idir)

	# Find the files for each window
	procE = 0
	while (isfile(dir+'/'+extMom_pre+'p'+str(procE)+extMom_suf)):
		procE += 1

	procL = 0
	while (isfile(dir+'/'+colMat_pre+'p'+str(procL)+colMat_suf)):
		procL += 1

	max_safe_proc = np.min([procL-1, procE-1])
	if (max_safe_proc < 1): raise Exception ('No windows found at all')

	lnPI_fname = [dir+'/'+colMat_pre+'p'+str(p)+colMat_suf for p in range(0, max_safe_proc)]
	mom_fname = [dir+'/'+extMom_pre+'p'+str(p)+extMom_suf for p in range(0, max_safe_proc)]
	
	return zip(lnPI_fname, mom_fname)

if __name__ == '__main__':
	print "feasst_patch.pyx"

	"""

	* Tutorial:

	1. Run get_patch_sequence() in simulation directory where each window is stored in a
		local directory labelled numerically (1, 2, 3,...)
	2. Feed the output to patch_all_windows() to patch together all results

	* Notes:

	If desired, an intermediate check can be applied between steps 1 and 2 to remove
		windows deemed "unequilibrated".  This is designed to operate with FEASST code
		by H. W. Hatch.

	"""
