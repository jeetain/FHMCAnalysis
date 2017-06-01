"""@docstring
@brief Library to estimate the quality of equilibration achieved by windows during a TMMC simulation using FEASST
@author Nathan A. Mahynski
@date 05/23/2017
@filename feasst_equil.pyx
"""

import os, sys, re, copy, cython, types
import numpy as np
import fhmc_patch as oP

cimport numpy as np
cimport cython

from os import listdir
from os.path import isfile, join
from numpy import ndarray
from numpy cimport ndarray
from cpython cimport bool

np.seterr(divide='raise', over='raise', invalid='raise') # Any sort of problem (except underflow), raise an exception

@cython.boundscheck(False)
@cython.cdivision(True)
cdef test_nebr_match_ (seq1, seq2, double per_err=1.0):
	"""
	Look at a neighboring pair of windows and see if extensive properties (besides lnPI) at same value of N_tot or N_1 (order parameter) are within a given error tolerance of each other.

	Requires that window1 < window2, else returns an error. This routine determines two neighbors are "converged" if the maximum deviation of the extensive properties in the overlapping region is below some threshold prescribed by the user.

	Parameters
	----------
	seq1 : tuple
		Filenames defining first window's information (colMat_fname, extMom_fname)
	seq2  : tuple
		Filenames defining first window's information (colMat_fname, extMom_fname)
	per_err : double
		Percent error considered tolerable for absolute value of deviation of all properties. If <= 0, all deviations are permissable (default=1.0)

	Returns
	-------
	bool, double, double
		match, max_u_err, max_n_err If neighbor pair is within specified bounds, max percent error from energy, max percent error from particle number

	"""

	cdef bool in_bounds = True # Flag to check if all values fall within the desired error tolerance
	max_per_err = np.inf
	combo_seq = [seq1, seq2]

	# Get some metadata
	nspec = [0,0]
	order = [0,0]
	for i in range(2):
		with open(combo_seq[i][1], 'r') as f:
			for line in f:
				if (line[0] == "#"):
					if ("maxOrder" in line):
						#info = line.strip().split(":")
						info = line.strip().split(" ")
						order[i] = int(info[len(info)-1])
					elif ("nSpec" in line):
						#info = line.strip().split(":")
						info = line.strip().split(" ")
						nspec[i] = int(info[len(info)-1])
				else:
					break

	assert (order[0] == order[1]), 'Different maximum orders found'
	assert (nspec[0] == nspec[1]), 'Different number of species found'

	# Find bounds to get overlap (assumes no offsets, so no data is disregarded at the ends)
	ub = [0, 0]
	lb = [0, 0]
	mom = []
	mom_exp = []
	for i in xrange(2):
		data = np.loadtxt(combo_seq[i][0], unpack=True)
		lb[i] = int(data[0][0])
		ub[i] = int(data[0][-1])

		# Get moments
		dummy_mom = np.loadtxt(combo_seq[i][1], dtype=np.float64, comments="#", unpack=False)
		mom.append(np.zeros(len(dummy_mom), dtype=np.float64))
		mom_exp.append(np.zeros((len(dummy_mom),5), dtype=np.float64))
		ctr = 0
		for row in dummy_mom:
			opIdx, nValues, Sum, SumSq, ii, jj, kk, mm, pp = row
			mom[i][ctr] = Sum/nValues
			mom_exp[i][ctr] = [ii,jj,kk,mm,pp]
			ctr += 1

	assert (ub[0] < ub[1]), 'Windows are out of order'
	assert (lb[0] < lb[1]), 'Windows are out of order'
	assert (ub[0] > lb[1]), 'Neighboring windows do not overlap'
	cdef int dw = ub[0] - lb[1] + 1

	# Test U
	uvals = []
	for i in xrange(2):
		idx = np.where((mom_exp[i] == [0,0,0,0,1]).all(axis=1))[0]
		assert (len(idx) == int(ub[i]-lb[i]+1)), 'Could not find energy entry for each value of the order parameter : '+str(len(idx))+' vs '+str(ub[i]-lb[i]+1)
		uvals.append(mom[i][idx])

	ov1 = uvals[0][len(uvals[0])-dw:]
	ov2 = uvals[1][:dw]
	assert (len(ov1) == len(ov2)), 'Bad overlap calculation'

	# Ideal gas check (U = 0?) because error is artificially inflated numerically if it is supposed to be 0, or nearly so
	tol = 1.0e-9
	max_u_err = -np.inf
	for i in xrange(len(ov1)):
		if (np.abs(ov1[i]) > tol):
			err = np.abs((ov1[i]-ov2[i])/ov1[i])*100.0
		elif (np.abs(ov2[i]) > tol):
			err = np.abs((ov1[i]-ov2[i])/ov2[i])*100.0
		else:
			err = -np.inf
		max_u_err = np.max([max_u_err, err])

	# Test N1, N2, ...
	ni = []
	for i in xrange(2):
		nj = []
		for j in xrange(nspec[0]):
			idx = np.where((mom_exp[i] == [j,1,0,0,0]).all(axis=1))[0]
			assert (len(idx) == int(ub[i]-lb[i]+1)), 'Could not find particle number entry for each value of the order parameter : '+str(len(idx))+' vs '+str(ub[i]-lb[i]+1)
			nj.append(mom[i][idx])
		ni.append(nj)

	max_n_err = 0.0
	for j in xrange(nspec[0]):
		ov1 = ni[0][j][len(ni[0][j])-dw:]
		ov2 = ni[1][j][:dw]
		assert (len(ov1) == len(ov2)), 'Bad overlap calculation'
		max_n_err = np.max([max_n_err, np.max(np.abs((ov2-ov1)/ov1))*100.0])

	ipass = False
	if (np.max([max_u_err, max_n_err]) < per_err):
		ipass = True

	return ipass, max_u_err, max_n_err

cpdef test_nebr_equil (seq, double per_err=3.0, fname='maxEq', bool trust=False):
	"""
	Tests equilibration of windows based on convergence of neighboring windows' extensive properties and returns a "safe" sequence of filenames for each window to use.

	This routine checks that windows are given ascending order, and returns the largest window whose larger neighbor is converged.  The largest window whose neighbor is NOT converged can be returned if the flag, endpoint is True.

	Parameters
	----------
	seq : list
		Sorted list of filename tuples (colMat_fname, extMom_fname) describing windows to look at
	per_err : double
		Max tolerable percent error in extensive thermodynamic properties between neighboring windows (default=3.0)
	fname : str
		File to write results to, can be read by other scripts since only non-commented block is maximum window to consider patching.  If 'None' then suppresses output. (default='maxEq')
	trust : bool
		If True, report the last window whose next neighbor is NOT converged, else report last window whose neighbor IS converged. (default=False)

	Returns
	-------
	array
		Refined sequence which is guaranteed continuous in window number up to the last window which is deemed equilibrated

	"""

	cdef int i, j, w, l_w, u_w
	cdef bool ipass, print_file = False, found = False
	cdef double max_u_err, max_n_err

	# double check that sequence is sorted by window index
	ordered_seq = []
	for i in xrange(len(seq)-1):
		if (i == 0):
			# Check the first sequence on the first execution of loop
			for j in xrange(0, len(seq[i])):
				x = seq[i][j].split('/')
				w = int(x[len(x)-2])
				if (j == 0):
					l_w = w
				else:
					assert (l_w == w), 'Window changes within sequence'
		else:
			# In subsequent loops, this one has already been checked because of code below
			l_w = u_w

		# Check the next one proposed
		for j in xrange(0, len(seq[i+1])):
			x = seq[i+1][j].split('/')
			w = int(x[len(x)-2])
			if (j == 0):
				u_w = w
			else:
				assert (u_w == w), 'Window changes within sequence'

		if (u_w == l_w+1):
			ordered_seq.append((seq[i],seq[i+1]))
		else:
			break

	if (fname != 'None'):
		print_file = True
		output = open(fname, 'w')
		output.write("#\tParameters used:\n")
		output.write("#\tpercent_err = "+str(per_err)+"\n")
		output.write("#\t(window i, window j)\tMax(%)_err\tMax(%U)_err\tMax(%N_i)_err")

	safe_seq = []
	for l_seq,u_seq in ordered_seq:
		ipass, max_u_err, max_n_err = test_nebr_match_ (l_seq, u_seq, per_err)
		if (ipass):
			found = True

			if (trust):
				if (len(safe_seq) == 0):
					safe_seq.append(l_seq)
				safe_seq.append(u_seq)
			else:
				safe_seq.append(l_seq)

			if (print_file):
				x = l_seq[0].split('/')
				w1 = int(x[len(x)-2])
				x = u_seq[0].split('/')
				w2 = int(x[len(x)-2])
				output.write('\n#\t('+str(w1)+','+str(w2)+')\t'+str(np.max([max_u_err, max_n_err]))+'\t'+str(max_u_err)+'\t'+str(max_n_err))
		else:
			break

	if (print_file):
		if (not found):
			output.close()
			raise Exception ('No safe windows found')
		else:
			if (trust):
				output.write('\n'+str(w2))
			else:
				output.write('\n'+str(w1))
			output.close()

	return safe_seq

if __name__ == "__main__":
	print "feasst_equil.pyx"

	"""

	* Tutorial:

	1. From feasst_patch, run seq = get_patch_sequence() in simulation directory where each window is stored in a local directory labelled numerically (1, 2, 3,...).
	2. Run new_seq = test_nebr_equil(seq) on sequence obtained from get_patch_sequence() to determine continuous sequence of windows up to max window.
	3. From feasst_patch, feed the output to patch_all_windows(new_seq) to patch together all results.

	* Notes:

	"""
