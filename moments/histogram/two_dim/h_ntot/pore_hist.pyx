"""@docstring
@brief Create a pore histogram based on simulations at fixed h, using Ntot as the order parameter
@author Nathan A. Mahynski									
@date 08/06/2016									
@filename pore_hist.pyx									
"""

import operator, sys, copy, cython, types
import numpy as np
import copy, json
sys.path.append('../')
import joint_hist as jh

cimport cython
cimport numpy as np

from numpy import ndarray
from numpy cimport ndarray
from cpython cimport bool
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport fabs

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries
from skimage.measure import profile_line

cdef inline double double_max (double a, double b): return a if a > b else b
cdef inline double double_min (double a, double b): return a if a < b else b

@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double specExp (double a, double b):
	"""
	Compute the natural logarithm of the sum of a pair of exponentials. i.e. ln(exp(a) + exp(b)) 
	
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
cdef np.ndarray[np.double_t, ndim=2] _cy_normalize (np.ndarray[np.double_t, ndim=2] data, np.ndarray[int, ndim=1] edge):
	"""
	Cythonized normalization intended for use with pore_hist. (pore_hist.normalize)

	Parameters
	----------
	data : ndarray
		Current lnPI(h,N) distribution
	edge : ndarray
		Array of indices corresponding to the end of the lnPI(N; h) distribution

	Returns
	-------
	ndarray
		normalized lnPI

	"""

	cdef double lnNormPI = -sys.float_info.max
	cdef int i, j
	for i in xrange(len(data)):
		for j in xrange(0, edge[i]+1):
			lnNormPI = specExp (lnNormPI, data[i,j])
	return data - lnNormPI

class pore_hist (object):
	"""
	A joint histogram in terms of (h, Ntot) from a general joint histogram object.
	
	Does not allow for reweighting.  This should be done with the raw data, before the joint_histogram
	object is constructed.

	"""
	
	def __init__(self, object joint_hist, object fh, double p_tot, double A, double beta):
		"""
		Instantiate the class.

		Parameters
		----------
		joint_hist : joint_hist
			Joint histogram in terms of (h, Ntot)
		fh : function
			F(h) function, free energy of empty adsorbent as a function of h (see free_energy_profile.pyx for examples)
		p_tot : double
			Total pressure, i.e., P_gas(beta, mu) + P_ext (For GC-ensemble, just set to 0)
		A : double
			Cross-sectional area of the slit-pore
		beta : double
			1/kT

		"""

		self.clear()
		self.data['F(h)'] = fh
		self.data['p'] = p_tot
		self.data['hist'] = copy.deepcopy(joint_hist)
		self.data['A'] = A
		self.data['beta'] = beta

		try:
			self.data['hist'].make()
		except Exception as e:
			raise Exception ('Could not construct joint histogram: '+str(e))

		# 0 <= N <= Nmax, continuous
		assert (np.all(self.data['hist'].data['op_2'] == np.arange(len(self.data['hist'].data['op_2'])))), 'Must be 0 <= N <= N_max in a continuous fashion'
		
		# lower bound all at 0 across the board, the upper bound defines ridgeline or 'edges'
		assert (np.all( [self.data['hist'].data['bounds_idx'][i,0] for i in xrange(len(self.data['hist'].data['op_1']))] == [0 for i in xrange(len(self.data['hist'].data['op_1']))] )), 'Lower bound for N must stazrt from 0'
		self.data['edge_idx'] = np.array([self.data['hist'].data['bounds_idx'][i,1] for i in xrange(len(self.data['hist'].data['op_1']))], dtype=int)
		self.data['mask'] = self.data['ln(PI)'] > -np.inf

		# create the pore histogram ln(PI) surface and normalize
		self.data['ln(PI)'] = copy.copy(self.data['hist'].data['ln(PI)'])
		for i in xrange(len(self.data['hist'].data['op_1'])):
			h = self.data['hist'].data['op_1'][i]
			shift = -self.data['beta']*(self.data['F(h)'](h) + self.data['p']*self.data['A']*h) - self.data['ln(PI)'][i,0]
			self.data['ln(PI)'][i,:] += shift
		self.normalize()

	def clear(self):
		"""
		Clear all data in histogram.

		"""

		self.data = {}
	
	def normalize(self):
		"""
		Normalize the ln(PI) surface.

		"""

		self.data['ln(PI)'] = _cy_normalize(self.data['ln(PI)'], self.data['edge_idx'])

	def thermo(self, mask):
		"""
		Compute the average extensive properties from a region of space.

		Parameters
		----------
		mask : ndarray
			Mask for lnPI which defines the region of space that belongs to the phase of interest

		Returns
		-------
		dict
			Dictionary of {property_name: average_value}

		"""

		cdef np.ndarray[np.double_t, ndim=2] lp = copy.deepcopy(self.data['ln(PI)'])
		lp -= np.max(lp) # shift for numerical stability
		lp[not mask] = -np.inf # set lnPI = -inf where we don't care		
		lp -= np.log(np.sum(np.exp(lp))) # normalize
		lp[not mask] = -np.inf # again for numerical security

		cdef np.ndarray[np.double_t, ndim=2] prob = np.exp(lp)
		cdef np.ndarray[np.double_t, ndim=2] sum_prob = np.sum(prob)

		ave_props = {}
		for prop in self.data['hist'].data['props']:
			ave_props[prop] = np.sum(prob*self.data['hist'].data['props'][prop])/sum_prob
		ave_props['peak_idx'] = np.where(lp == np.max(lp))
		
		return ave_props

	def phase_average(self, int nnebr=1, int max_peaks=10):
		"""
		Get average properties of regions of lnPI (h, N) state space after considering only their local maxima in this space (only points on same "hill").	

		Parameters
		----------									
		nnebr : int
			Number of neighbor positions in unscaled (h, N) space to smooth over (default=1)
		max_peaks : int 
			Maximum number of peaks to identify (default=10)

		"""

		cdef int j, n, i, ctr, hill
		cdef double ln_f = -sys.float_info.max, pore_cutoff = 10.0
		
		self.normalize()
		max_peaks += 1 # to account for background
		try:
			self._segment(nnebr, max_peaks)
		except Exception as e:
			raise Exception ('Cannot segment the surface: '+str(e))

		uniqueMax, uniqueCounts = np.unique (self.data['seg']['phase_labels'], return_counts=True)
		for j in xrange(0, len(self.data['ln(PI)'])):
			ln_f = specExp (ln_f, self.data['ln(PI)'][j,0])

		# calculate free energy along the transition state path
		self.data['seg']['transition_state_kT'][self.data['seg']['transition_state_kT'] > -sys.float_info.max] -= ln_f
		self.data['seg']['transition_state_kT'][self.data['seg']['transition_state_kT'] > -sys.float_info.max] *= -1.0

		phase_props = {}
		ctr = 0
		for i in xrange(len(uniqueMax)):
			hill = uniqueMax[i]
			if hill < 1: # skip points where lnPI = -np.inf (not sampled/extrapolated, and not tested in steepest ascent)
				continue
			
			mask = self.data['seg']['phase_labels'] == hill
			ave_props = self.thermo(mask)
			ave_props['F.E./kT'] = ln_f - np.log(np.sum(np.exp(self.data['ln(PI)'][mask])))
			phase_props[ctr] = copy.deepcopy(ave_props)
			ctr += 1

			# check diff between max(lnPI) and largest ridgeline value in this phase
			ridge_vals = [self.data['ln(PI)'][h,self.data['edge_idx']] if mask[h,self.data['edge_idx'][h]] == True else -np.inf for h in xrange(0, len(self.data['edge_idx']))] 
			max_diff = np.max(self.data['ln(PI)'][mask]) - np.max(ridge_vals) 
			if (max_diff < pore_cutoff):
				raise Exception ("Cannot compute phase_average because of ridgeline effects")

		# calculate activation free energy
		act_kT = np.zeros((len(uniqueMax)-1, len(uniqueMax)-1), dtype=np.float64)
		act_kT_diff = np.zeros((len(uniqueMax)-1, len(uniqueMax)-1), dtype=np.float64)
		for i in xrange(1, len(uniqueMax)):
			for j in xrange(i+1, len(uniqueMax)):
				if (self.data['seg']['transition_state_kT'][i,j] > -sys.float_info.max):
					act_kT[i-1,j-1] = self.data['seg']['transition_state_kT'][i,j]
					act_kT[i-1,j-1] -= double_max (phase_props[i-1]['F.E./kT'], phase_props[j-1]['F.E./kT']) # largest free energy is least stable of the two
					act_kT[j-1,i-1] = act_kT[i-1,j-1]

					act_kT_diff[i-1][j-1] = double_min(self.data[self.data['seg']['local_maxima'][i-1,0]][self.data['seg']['local_maxima'][i-1,1]],self.data[self.data['seg']['local_maxima'][j-1,0]][self.data['seg']['local_maxima'][j-1,1]])-self.data['seg']['max_border_kT'][i,j]
					act_kT_diff[j-1,i-1] = act_kT_diff[i-1,j-1]

		phase_props['activation_kT'] = act_kT # does not include 'background'			
		phase_props['activation_kT_diff'] = act_kT_diff # does not include 'background'

		return phase_props

	def width_phase_average(self, h_divide, int nnebr=1, int max_peaks=10):
		"""
		Perform phase_average calculation to obtain properties of each phase, but then post-process to 
		consider them as "subphases" within a superclass, i.e. h < h_transition, that defines a
		macroscopic phase.  


		For example, phase_average could find several maxima within what the user may define as a "narrow-pore" phase.  Rather than considering each of those maxima as independent, this will collect/merge them into a single phase and average over that (h,N) space defined by the h_bounds. This can be used to average over the entire distribution by setting h_divide > h_max.

		Parameters
		----------
		h_divide : ndarray 
			h values that divide(s) phase(s) (does not need to be sorted, but will be later to define which phase is which)							
		nnebr : int
			Number of neighbor positions in unscaled (h, N) space to smooth over (default=1)
		max_peaks : int
			Maximum number of peaks to identify (default=10)

		"""

		cdef int j, n, i, hill
		cdef double ln_f = -sys.float_info.max, pore_cutoff = 10.0
		
		h_divide = sorted(h_divide)
		assert (max_peaks > len(h_divide)), 'Cannot create that many phases when expecting less local maxima in ln(PI)'
		
		self.normalize()
		max_peaks += 1 # to account for background
		try:
			self._segment(nnebr, max_peaks)
			assign = self._collect(h_divide)
		except Exception as e:
			raise Exception ('Cannot segment the surface: '+str(e))

		for j in xrange(0, len(self.data['ln(PI)'])):
			ln_f = specExp (ln_f, self.data['ln(PI)'][j,0])
		
		# calculate free energy along the transition state path
		self.data['seg']['transition_state_kT'][self.data['seg']['transition_state_kT'] > -sys.float_info.max] -= ln_f
		self.data['seg']['transition_state_kT'][self.data['seg']['transition_state_kT'] > -sys.float_info.max] *= -1.0

		phase_props = {}
		for i in sorted([ph for ph in assign]):
			mask = []
			assert(len(assign[i]) > 0), 'Width-defined phase does not contain any local maxima in ln(PI)'
			for hill in assign[i]:
				if (mask == []):
					mask = (self.data['seg']['phase_labels'] == hill)
				else:
					mask = mask | (self.data['seg']['phase_labels'] == hill)
			
			ave_props = self.thermo(mask)
			ave_props['F.E./kT'] = ln_f - np.log(np.sum(np.exp(self.data['ln(PI)'][mask])))
			phase_props[i] = copy.deepcopy(ave_props)

			# check diff between max(lnPI) and largest ridgeline value in this phase
			ridge_vals = [self.data['ln(PI)'][h,self.data['edge_idx']] if mask[h,self.data['edge_idx'][h]] == True else -np.inf for h in xrange(0, len(self.data['edge_idx']))] 
			max_diff = np.max(self.data['ln(PI)'][mask]) - np.max(ridge_vals) 
			if (max_diff < pore_cutoff):
				raise Exception ("Cannot compute phase_average because of ridgeline effects")

		# for now, no activation energy associated
		
		return phase_props

	def _collect(self, np.ndarray[np.double_t, ndim=1] h_divide):
		"""
		After segmentation into individual phases, collect into "super-phases" based on h values
	
		Parameters
		----------
		h_divide : ndarray
			h value(s) that divide(s) phases (does not need to be sorted)

		Returns
		-------
		dict
			Dictionary of {phase_idx: [hills that belong]}

		"""

		h_div = sorted(h_divide)
		h_idx = np.zeros(len(h_div), dtype=int)
		
		cdef int i, h_ctr = 0
		for i in xrange(len(self.data['hist'].data['op_1'])):
			h = self.data['hist'].data['op_1'][i]
			if (h > h_div[h_ctr]):
				h_idx[h_ctr] = i-1
				h_ctr += 1
		
		if (h_ctr == len(h_divide)-1):
			h_idx[h_ctr] = len(self.data['hist'].data['op_1'])-1
		elif (h_ctr < len(h_divide)-1):
			raise Exception ('Unable to divide h-space')
		
		assign = {}
		uniqueMax, uniqueCounts = np.unique (self.data['seg']['phase_labels'], return_counts=True)
		for i in xrange(len(uniqueMax)):
			hill = uniqueMax[i]
			if hill < 1: # skip points where lnPI = -np.inf (not sampled/extrapolated, and not tested in steepest ascent)
				continue
			
			mask = self.data['seg']['phase_labels'] == hill
			tmp = copy.deepcopy(self.data['ln(PI)'])
			tmp[not mask] = -np.inf
			h_loc = np.where(tmp == np.max(tmp))[0][0]
		
			phase = 0
			while (h_loc > h_idx[phase]):
				phase += 1

			if (phase not in assign):
				assign[phase] = [hill]
			else:
				assign[phase].append(hill)
		
		for phase in xrange(0, len(h_idx)):
			if phase not in assign:
				assign[phase] = []
		
		return assign
	
	def _segment(self, int nnebr=1, int num_peaks=10):
		"""
		Segment the surface into hills and basins using a "steepest ascent" climb on discrete lnPI (h, N) surface.		

		The number of neighbors to look over, nnebr, should not be too large otherwise will lose precision near the saddle point between extrema. Peaks are separated by atleast 2*nnebr + 1 pixels. The footprint is scaled to 'equalize' dimensions.		

		Parameters
		----------									
		nnebr : int 
			Number of neighbor positions in unscaled (h, N) space to smooth over (default=1)
		num_peaks : int
			Maximum number of peaks to identify (default=10)

		"""

		self.data['seg'] = {}

		# using watershed
		cdef double scale_h, scale_n, dh_coord, dn_coord
		cdef double n_incrs = float(len(self.data['hist'].data['op_2'])-1), h_incrs = float(len(self.data['hist'].data['op_1'])-1), ave_val
		cdef int len_N = len(self.data['hist'].data['op_2']), len_H = len(self.data['hist'].data['op_1']), i, j, this_phase, nebr_phase

		sd = self.data['ln(PI)']

		if (h_incrs >= n_incrs):
			scale_h = 1.0
			scale_n = h_incrs/n_incrs
		else:
			scale_h = n_incrs/h_incrs
			scale_n = 1.0

		cdef int fp_x = np.round(scale_n*nnebr)*2+1, fp_y = np.round(scale_h*nnebr)*2+1, bnd_x = (fp_x-1)/2-1, bnd_y = (fp_y-1)/2-1
		footprint = np.ones((fp_x, fp_y))

		# shift the pixel data to be => 0
		cdef np.ndarray[np.double_t, ndim=2] x = sd - np.min(sd[self.data['mask']])
		x[self.data['mask']] = 0
		self.data['seg']['local_maxima'] = peak_local_max (x, min_distance=nnebr, exclude_border=0, num_peaks=num_peaks, indices=True, footprint=footprint)
		cdef int n_maxima = len(self.data['seg']['local_maxima'])
		lm = self.data['seg']['local_maxima']

		cdef np.ndarray[int, ndim=2] markers = np.zeros((len_H, len_N), dtype=int)
		for i in xrange(n_maxima):
			markers[lm[i][0]][lm[i][1]] = i+1

		# watershed to segment
		ans = watershed(-x, markers=markers, mask=self.data['mask'], connectivity=footprint)
		self.data['seg']['phase_labels'] = ans[bnd_x:-bnd_x, bnd_y:-bnd_y] # trim edges left over from 'wide' footprint

		# find boundaries and integrate lnPI along boundary
		cdef np.ndarray[np.double_t, ndim=2] min_df = np.empty((n_maxima+1, n_maxima+1), dtype=np.float64), max_val = np.empty((n_maxima+1, n_maxima+1), dtype=np.float64)
		min_df.fill(-sys.float_info.max)	
		max_val.fill(-sys.float_info.max)
		my_edges = find_boundaries(self.data['seg']['phase_labels'], connectivity=1, mode='inner', background=0)		
		ix,iy = np.where(my_edges)
		pl = self.data['seg']['phase_labels']
		nebr_vecs = [[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,1]] 
		for i,j in zip(ix,iy):
			this_phase = pl[i][j]
			for k,m in nebr_vecs:
				if (i+k >= 0 and i+k < len_H and j+m >= 0 and j+m < len_N):
					nebr_phase = pl[i+k,j+m]
					if (nebr_phase != this_phase and nebr_phase > 0 and this_phase > 0): # transitions between different phases which are not background
						ave_val = specExp (sd[i,j]-log(2.0), sd[i+k,j+m]-log(2.0))
						min_df[this_phase,nebr_phase] = specExp(min_df[this_phase,nebr_phase], ave_val)
						min_df[nebr_phase,this_phase] = min_df[this_phase,nebr_phase]
						max_val[this_phase,nebr_phase] = double_max(max_val[this_phase,nebr_phase], ave_val)
						max_val[nebr_phase,this_phase] =  max_val[this_phase,nebr_phase]

		self.data['seg']['transition_state_kT'] = min_df # will be converted to a free energy in phaseAverage
		self.data['seg']['max_border_kT'] = max_val	

		# Make lines between maxima using (0,0) and (h,N)_max as the reference endpoints
		start = []
		end = []
		start.append((0,0))
		order = np.lexsort((lm[:,1], lm[:,0]))
		for i in xrange(len(lm)):
			start.append((lm[order][i][0],lm[order][i][1]))
			end.append((lm[order][i][0],lm[order][i][1]))
		end.append((len_H, len_N))
		
		# Get profile along each line
		line_profile = np.array([])
		line_profile_coords = []
		
		for i in xrange(len(start)):
			intensity = profile_line(x, start[i], end[i], linewidth=1, order=0, cval=0.0)
			dh_coord = (end[i][0] - start[i][0])/float(len(intensity))
			dn_coord = (end[i][1] - start[i][1])/float(len(intensity))			
			if (i == 0):
				line_profile = np.concatenate ((line_profile, intensity)) 
				for j in xrange(0, len(intensity)):
					line_profile_coords.append([start[i][0] + dh_coord*j, start[i][1] + dn_coord*j])
			else:
				line_profile = np.concatenate ((line_profile, intensity[1:])) # trim first point (last end) since was included already, see profile_line documentation
				for j in xrange(1, len(intensity)):
					line_profile_coords.append([start[i][0] + dh_coord*j, start[i][1] + dn_coord*j])

		self.data['seg']['line_profile'] = line_profile + np.min(sd[self.data['mask']]) # shift lnPI back to real values (not 'intensities')
		self.data['seg']['line_profile_coords'] = np.array(line_profile_coords)
	
if __name__ == '__main__':
	print 'pore_hist.pyx'
	
	"""
	
	* Tutorial:
	
	1.) Make each gc_hist and reweight/extrapolate as necessary
	2.) Combine into joint_hist
	3.) Use joint_hist to produce pore_hist and analyze
	
	* Notes:
	
	* To Do:
	
	Finish width_phase_average
	Add ridgeline and h-bounds checks for phase_average() calculation to ensure no boundary effects
	
	"""
