"""@docstring
@brief Library to estimate the quality of equilibration achieved by windows during a TMMC simulation
@author Nathan A. Mahynski
@date 07/19/2016
@filename omcs_equil.pyx
"""

import os, sys, re, copy, cython, types
import numpy as np
import omcs_patch as oP

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
cpdef test_nebr_match (seq1, seq2, double per_err=1.0):
	"""
	Look at a neighboring pair of windows and see if extensive properties (besides lnPI) at same value of N_tot (order parameter) are within a given error tolerance of each other.

	Requires that window1 < window2, else returns an error. This routine determines two neighbors are "converged" if the maximum deviation of the extensive properties in the overlapping region is below some threshold prescribed by the user.

	Parameters
	----------
	seq1 : tuple
		Filenames defining first window's information (lnPI_fname, mom_fname, ehist_fname, pkhist_prefix)
	seq2  : tuple
		Filenames defining first window's information (lnPI_fname, mom_fname, ehist_fname, pkhist_prefix)
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

	# Find bounds to get overlap (assumes no offsets, so no data is disregarded at the ends)
	ub = [0, 0]
	lb = [0, 0]
	for i in xrange(2):
		f = open(combo_seq[i][0])
		data = f.readline() # description line
		data = f.readline() # ub
		data = re.split("_|:|\\n| |",data)
		ub[i] = int(data[len(data)-2])
		data = f.readline() # lb
		data = re.split("_|:|\\n| |",data)
		lb[i] = int(data[len(data)-2])
		f.close()

	assert (ub[0] < ub[1]), 'Windows are out of order'
	assert (lb[0] < lb[1]), 'Windows are out of order'
	assert (ub[0] > lb[1]), 'Neighboring windows do not overlap'
	cdef int dw = ub[0] - lb[1] + 1

	# Test U
	max_order = [0,0]
	nspec = [0,0]
	uvals = []
	for i in xrange(2):
		info = np.loadtxt(combo_seq[i][1], unpack=True)
		f = open(combo_seq[i][1], 'r')
		f.readline()
		data = f.readline()
		data = re.split("_|:|\\n| |",data)
		nspec[i] = int(data[len(data)-2])
		data = f.readline()
		data = re.split("_|:|\\n| |",data)
		max_order[i] = int(data[len(data)-2])
		f.close()
		assert (max_order[i] >= 1), 'Must record atleast 1st moment to get average property'
		uvals.append(info[2,:])

	assert (max_order[0] == max_order[1]), 'Different maximum order in each window'
	assert (nspec[0] == nspec[1]), 'Different number of species in each window'
	ov1 = uvals[0][len(uvals[0])-dw:]
	ov2 = uvals[1][:dw]
	assert (len(ov1) == len(ov2)), 'Bad overlap calculation'

	# ideal gas check (U = 0?)
	max_u_err = -np.inf
	for i in xrange(len(ov1)):
		if (ov1[i] != 0.0):
			err = np.abs((ov1[i]-ov2[i])/ov1[i])*100.0
		elif (ov2[i] != 0.0):
			err = np.abs((ov1[i]-ov2[i])/ov2[i])*100.0
		else:
			err = -np.inf
		max_u_err = np.max([max_u_err, err])

	# Test N1, N2, ...
	ni = []
	mo = max_order[0]+1
	for i in xrange(2):
		info = np.loadtxt(combo_seq[i][1], unpack=True)
		nj = []
		for j in xrange(nspec[0]):
			address = 1 + (0 + mo*0 + mo*mo*0 + mo*mo*nspec[0]*1 + mo*mo*nspec[0]*mo*j)
			nj.append(info[address,:])
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

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef test_window_match (win1_dir, win2_dir, double per_err=1.0, int min_cp=-1):
	"""
	Look at a neighboring pair of windows and see if extensive properties (besides lnPI) at same value of N_tot (order parameter) are within a given error tolerance of each other.

	Requires that window1 < window2, else returns an error. This routine determines two neighbors are "converged" if the maximum deviation of the extensive properties in the overlapping region is below some threshold prescribed by the user.

	Parameters
	----------
	win1_dir : str
		First window directory
	win2_dir : str
		Second window directory
	per_err : double
		Percent error considered tolerable for absolute value of deviation of all properties. If <= 0, all deviations are permissable (default=1.0)
	min_cp : int
		Minimum checkpoint required for consideration (default=-1, i.e., no requirement)

	Returns
	-------
	bool, double, double
		match, max_u_err, max_n_err If neighbor pair is within specified bounds, max percent error from energy, max percent error from particle number

	"""

	cdef bool in_bounds = True # Flag to check if all values fall within the desired error tolerance
	max_per_err = np.inf

	lnPI_fname = []
	mom_fname = []
	ehist_fname = []
	pkhist_prefix = []
	win_dirs = [win1_dir, win2_dir]

	# Get filenames to compare (latest results available)
	for i in xrange(2):
		files = listdir(win_dirs[i])
		d = win_dirs[i]

		# look for the final information, whatever that is
		if ("final_lnPI.dat" in files):
			lnPI_fname.append(d+"/final_lnPI.dat")
			mom_fname.append(d+"/final_extMom.dat")
			ehist_fname.append(d+"/final_eHist.dat")
			pkhist_prefix.append(d+"/final_pkHist")
		else:
			l = []
			m = []
			p = []
			q = []
			min_cp_reached = np.inf
			found = {'tmmc':False, 'mom':False, 'eh':False, 'ph':False}
			for f in files:
				if ("tmmc-Checkpoint-" in f and ("_lnPI.dat" in f)):
					l.append(f)
					found['tmmc'] = True
					checkpt = re.split("_|-|\.",f)[2]
					min_cp_reached = np.min([min_cp_reached, int(checkpt)])
				if ("extMom-Checkpoint-" in f and (".dat" in f)):
					m.append(f)
					found['mom'] = True
					checkpt = re.split("_|-|\.",f)[2]
					min_cp_reached = np.min([min_cp_reached, int(checkpt)])
				if ("eHist-Checkpoint-" in f and (".dat" in f)):
					p.append(f)
					found['eh'] = True
					checkpt = re.split("_|-|\.",f)[2]
					min_cp_reached = np.min([min_cp_reached, int(checkpt)])
				if ("pkHist-Checkpoint-" in f and ("_1.dat" in f)): # only look for species 1
					q.append(f)
					found['ph'] = True
					checkpt = re.split("_|-|\.",f)[2]
					min_cp_reached = np.min([min_cp_reached, int(checkpt)])

			if (np.all([found[i] for i in found]) and min_cp_reached >= min_cp):
				oP.sort_nicely(l)
				oP.sort_nicely(m)
				oP.sort_nicely(p)
				oP.sort_nicely(q)

				lnPI_fname.append(d+"/"+l[len(l)-1])
				mom_fname.append(d+"/"+m[len(m)-1])
				ehist_fname.append(d+"/"+p[len(p)-1])
				pkhist_prefix.append(d+"/"+q[len(q)-1].split('_')[0])

	# Find bounds to get overlap (assumes no offsets, so no data is disregarded at the ends)
	ub = [0, 0]
	lb = [0, 0]
	for i in xrange(2):
		f = open(lnPI_fname[i])
		data = f.readline() # description line
		data = f.readline() # ub
		data = re.split("_|:|\\n| |",data)
		ub[i] = int(data[len(data)-2])
		data = f.readline() # lb
		data = re.split("_|:|\\n| |",data)
		lb[i] = int(data[len(data)-2])
		f.close()

	assert (ub[0] < ub[1]), 'Windows are out of order'
	assert (lb[0] < lb[1]), 'Windows are out of order'
	assert (ub[0] > lb[1]), 'Neighboring windows do not overlap'
	cdef int dw = ub[0] - lb[1] + 1

	# Test U
	max_order = [0,0]
	nspec = [0,0]
	uvals = []
	for i in xrange(2):
		info = np.loadtxt(mom_fname[i], unpack=True)
		f = open(mom_fname[i], 'r')
		f.readline()
		data = f.readline()
		data = re.split("_|:|\\n| |",data)
		nspec[i] = int(data[len(data)-2])
		data = f.readline()
		data = re.split("_|:|\\n| |",data)
		max_order[i] = int(data[len(data)-2])
		f.close()
		assert (max_order[i] >= 1), 'Must record atleast 1st moment to get average property'
		uvals.append(info[2,:])

	assert (max_order[0] == max_order[1]), 'Different max_order in each window'
	assert (nspec[0] == nspec[1]), 'Different number of species in each window'
	ov1 = uvals[0][len(uvals[0])-dw:]
	ov2 = uvals[1][:dw]
	assert (len(ov1) == len(ov2)), 'Bad overlap calculation'
	max_u_err = np.max(np.abs((ov2-ov1)/ov1))*100.0

	# Test N1, N2, ...
	ni = []
	mo = max_order[0] + 1
	for i in xrange(2):
		info = np.loadtxt(mom_fname[i], unpack=True)
		nj = []
		for j in xrange(nspec[0]):
			address = 1 + (0 + mo*0 + mo*mo*0 + mo*mo*nspec[0]*1 + mo*mo*nspec[0]*mo*j)
			nj.append(info[address,:])
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

cpdef find_windows (idir):
	"""
	Find directories of all windows, in order, which have completed and neighbor sets.

	Windows must be integer-labeled directories starting from 1.  Determines progress of simulation by checking for tmmc-Checkpoints-*_lnPI.dat files.  Window must complete at least 2 checkpoints.

	Parameters
	----------
	idir : str
		Head directory to search for windows in

	Returns
	-------
	array, array
		Array of window directories relative to head, Array of tuples listing neighboring windows by (n-1, n)

	"""

	cdef int max_cp, min_cp, ub, i
	windows = []
	nebr_set = []
	passed = []

	if (idir[len(idir)-1] == '/'):
		dir = idir[:len(idir)-1]
	else:
		dir = copy.copy(idir)

	# List all files/folders in dir and sort
	win_dir = [ f for f in listdir(dir) if not isfile(join(dir+"/",f)) ]
	for d in win_dir:
		files = listdir(dir+"/"+d)
		cp = [fi for fi in files if ("tmmc-Checkpoint-" in fi and "_lnPI.dat" in fi)] # Search based on tmmc-Checkpoints
		max_cp = 0
		for c in cp:
			max_cp = np.max([max_cp, int(re.split("_|-",c)[2])])
		if (max_cp >= 1):
			passed.append(int(d))
	passed = sorted(passed)

	# Starting from minimum, move up to file to ensure continuous sequence
	ub = passed[0]
	for i in xrange(1, len(passed)):
		if (passed[i] - passed[i-1] == 1):
			ub += 1
		else:
			break
	windows = np.arange(passed[0], ub+1)

	# Create neighbor set from continuous windows
	for i in xrange(windows[0], windows[len(windows)-1]):
		nebr_set.append((i, i+1))

	return windows, nebr_set

cpdef test_nebr_equil (seq, double per_err, fname='maxEq', bool trust=False):
	"""
	Tests equilibration of windows based on convergence of neighboring windows' extensive properties and returns a "safe" sequence of filenames for each window to use.

	This routine checks that windows are given ascending order, and returns the largest window whose larger neighbor is converged.  The largest window whose neighbor is NOT converged can be returned if the flag, endpoint is True.

	Parameters
	----------
	seq : list
		Sorted list of filename tuples (lnPI_fname, mom_fname, ehist_fname, pkhist_prefix) describing windows to look at
	per_err : double
		Max tolerable percent error in extensive thermodynamic properties etween neighboring windows
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
			for j in xrange(0, len(seq[i])):
				x = seq[i][j].split('/')
				w = int(x[len(x)-2])
				if (j == 0):
					l_w = w
				else:
					assert (l_w == w), 'Window changes within sequence'
		else:
			l_w = u_w

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
		ipass, max_u_err, max_n_err = test_nebr_match (l_seq, u_seq, per_err)
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
	print "omcs_equil.pyx"

	"""
	* Tutorial:

	1. From omcs_patch, run get_patch_sequence() in simulation directory where each window is stored in a local directory labelled numerically (1, 2, 3,...)
	2. Run test_nebr_equil() on sequence obtained from get_patch_sequence() to determine continuous sequence of windows up to max window
	3. From omcs_patch, feed the output to patch_all_windows() to patch together all results

	* Notes:

	In the future oP.alphanum_key() can be used to more generally identify numbers in strings.  This would make metadata parsing easier as well as checkpoint number identification more straightforward.
	"""
