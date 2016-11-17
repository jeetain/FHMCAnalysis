"""@docstring
@brief Organize "phases" during contiguous reweighting sequences for a pore (mu1 from -inf --> +inf)
@author Nathan A. Mahynski									
@date 09/01/2016									
@filename organize.pyx									
"""

import operator, sys, copy, cython, types
import numpy as np
import copy, json

cimport cython
cimport numpy as np

from numpy import ndarray
from numpy cimport ndarray
from cpython cimport bool
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport fabs
from libc.math cimport sqrt

class phase_organizer (object):
	"""
	After reweighting, this class organizes thermodynamic properties after each step into a self-consistent tracker for each phase.

	Peaks must move max(dH, dN) from last measurement to be considered a new phase.
	dH, dN should be consistent with scales used in steepestAscent. There I am expecting that
	peaks are atleast separated by 2*nNeighbors+1 pixels, so this should be atleast as large.
	If phases begin to converge, steepestAscent might find peaks converging and moving fast so
	this class will warn and exit if it receives different peaks but tries to assign to the same
	internally hashed phase.

	"""

	def __init__ (self, axes_ratio, nPix, max_phases):
		""" 
		Instatiate the class

		Parameters
		----------
		axes_ratio : double
			Ratio of number of x,y pixels.  For lnPI(h, N), this #h_bins/#n_bins (usually < 1)
		nPix : int
			Number of pixels (in 'x' direction = h) to enclose in a phase
		max_phases : int
			Maximum number of phases expected

		"""

		self.axes_ratio = axes_ratio
		self.nPix = nPix
		self.rcut2 = nPix**2
		self.phase_data = []
		self.last_pt = []
		self.dF_kT = []
		self.dF_kT_diff = []
		self.max_phases = max_phases
		self.max_err = 0.0

	def add (self, info):
		"""
		Add new properties found.

		Properties
		----------
		info : tuple
			(_mu1, _P, _phaseNtot, _phaseX, _phaseU, _phaseFreeEnergy, _phasePt, _phaseAveH, _phaseAct, _phaseActDiff) for all phases

		"""

		# First translate results to be self-consistent with previously recorded data
		translation = {}
		mu1, P, _phaseNtot, _phaseX, _phaseU, _phaseFreeEnergy, _phasePt, _phaseAveH, _phaseAct, _phaseActDiff = info
		for phase in xrange(0, len(_phasePt)):
			used = {}
			if (_phaseFreeEnergy[phase] != np.inf and _phasePt[phase] != []):
				idx = self.get_phase (_phasePt[phase])
				assert (idx < self.max_phases), "Too many phases ("+str(idx)+") have appeared for phase_organizer to handle (max = "+str(self.max_phases)+")"
				if idx in used:
					raise Exception ("Phase organizer wants to assign different calculated phases to same internally stored phase, try reducing rcut and increasing max_phases")
				else:				
					used[idx] = 1
					translation[phase] = idx
	
		# Translate transition states
		dF_kT = np.zeros((self.max_phases, self.max_phases), dtype=np.float64)
		dF_kT_diff = np.zeros((self.max_phases, self.max_phases), dtype=np.float64)	
		for phase1 in xrange(0, len(_phaseAct)):
			for phase2 in xrange(phase1+1, len(_phaseAct)):
				if (phase1 in translation and phase2 in translation):
					dF_kT[translation[phase1]][translation[phase2]] = _phaseAct[phase1][phase2]
					dF_kT[translation[phase2]][translation[phase1]] = _phaseAct[phase2][phase1]
					dF_kT_diff[translation[phase1]][translation[phase2]] = _phaseActDiff[phase1][phase2]
					dF_kT_diff[translation[phase2]][translation[phase1]] = _phaseActDiff[phase2][phase1]

		# Record
		for phase in translation:
			self.add_data ((mu1, P, _phaseNtot[phase], _phaseX[phase], _phaseU[phase], _phaseFreeEnergy[phase], _phasePt[phase], _phaseAveH[phase], dF_kT[translation[phase]], dF_kT_diff[translation[phase]]), translation[phase])
	
	def add_data (self, info, phase_idx):
		"""
		Organize new properties found.

		Parameters
		----------
		info : tuple
			(_mu1, _P, _phaseNtot, _phaseX, _phaseU, _phaseFreeEnergy, _phasePt, _phaseAveH, _dF_kT, _dF_kT_diff) for one phase
		phase_idx : int
			Internally hashed index this phase corresponds to

		"""	

		assert (phase_idx < self.max_phases), "Too many phases ("+str(phase_idx)+") have been identified for phase_organizer to handle (max = "+str(self.max_phases)+")"
		if (len(self.phase_data) > phase_idx):
			self.phase_data[phase_idx].append(info)
		else:
			self.phase_data.append([info])
		
	def get_phase (self, phasePt):
		"""
		Identifies internal index a phase corresponds to.  

		Adds phases as necessary up max_phases, then returns phase it is closest to.	

		Parameters
		----------
		phasePt : array
			[h,N] coordinate local maxima for phase

		Returns
		-------
		int
			Internally hashed index this corresponds to

		"""

		last_pt = self.last_pt
		if (len(self.last_pt) == 0):
			self.last_pt.append(phasePt)
			return 0
		else:
			idx = 0
			d2 = np.inf
			for i in xrange(0, len(self.last_pt)):
				h,N = self.last_pt[i]
				dist2 = ((self.last_pt[i][0] - phasePt[0]))**2 + ((self.last_pt[i][1] - phasePt[1])*self.axes_ratio)**2
				if (dist2 < d2):
					idx = i
					d2 = dist2

			if (d2 > self.rcut2): # new phase?
				if (len(self.last_pt) < self.max_phases): # allowed to create new?
					self.last_pt.append(phasePt)
					return len(self.last_pt)-1
				else:
					self.max_err = max(self.max_err, sqrt(d2)) # had to guess
					self.last_pt[idx] = phasePt
					return idx # return closest one found if at limit
			else:
				self.last_pt[idx] = phasePt
				return idx

	def print_org (self, prefix, comments=''):
		"""
		Print results to a file.	

		Parameters
		----------
		prefix : str
			File will be called prefix.json

		"""

		max_observed_phase = len(self.last_pt)
		f = open(prefix+".json", 'w')
		obj = {'Comments': comments, 'Max Guessing Err': self.max_err} # Error in best guess
		for i in xrange(0, len(self.phase_data)):
			nt = [self.phase_data[i][j][2] for j in xrange(0, len(self.phase_data[i]))]
			ut = [self.phase_data[i][j][4] for j in xrange(0, len(self.phase_data[i]))]
			fe = [self.phase_data[i][j][5] for j in xrange(0, len(self.phase_data[i]))]
			hv = [self.phase_data[i][j][7] for j in xrange(0, len(self.phase_data[i]))]
			idx = [[self.phase_data[i][j][6][0], self.phase_data[i][j][6][1]] for j in xrange(0, len(self.phase_data[i]))]
			mu = [self.phase_data[i][j][0] for j in xrange(0, len(self.phase_data[i]))]
			pp = [self.phase_data[i][j][1] for j in xrange(0, len(self.phase_data[i]))]
			xt = [self.phase_data[i][j][3].tolist() for j in xrange(0, len(self.phase_data[i]))]
			df = [self.phase_data[i][j][8][:max_observed_phase].tolist() for j in xrange(0, len(self.phase_data[i]))]
			df_d = [self.phase_data[i][j][9][:max_observed_phase].tolist() for j in xrange(0, len(self.phase_data[i]))]
			info = {'Phase': i, 'mu_1': mu, 'P': pp, 'N_tot': nt, 'U_tot': ut, 'Free_energy/kT': fe, '<h>': hv, 'x_i': xt, '(h,N)': idx, 'dF^t_i,j(integral)': df, 'dF^t_i,j(diff)': df_d}	
			obj[i] = info
		json.dump(obj, f, sort_keys=True, indent=4)	
		f.close()

if __name__ == '__main__':
	print 'organize.pyx'
	
	"""
	
	* Tutorial:
	
	* Notes:
	
	* To Do:

	"""
