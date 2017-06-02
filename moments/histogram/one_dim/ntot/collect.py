"""@docstring
@brief Functions to "collect" local maxima from segmented ln(PI) into "macrophases" when each individual peak does not correspond to a thermodynamic phase.
@author Nathan A. Mahynski
@date 06/02/2017
@filename collect.py
"""

import numpy as np

def check_order_ (hist):
	"""
	Check that the order alternates correctly.

	Parameters
        ----------
        hist : histogram
                Histogram that has been already segmented

	"""

        order = np.zeros(len(hist.data['ln(PI)_maxima_idx']) + len(hist.data['ln(PI)_minima_idx']))
        if (hist.data['ln(PI)_maxima_idx'][0] < hist.data['ln(PI)_minima_idx'][0]):
                order[::2] = hist.data['ln(PI)_maxima_idx']
                order[1::2] = hist.data['ln(PI)_minima_idx']
        else:
                order[::2] = hist.data['ln(PI)_minima_idx']
                order[1::2] = hist.data['ln(PI)_maxima_idx']

	if (not (np.all([order[i] <= order[i+1] for i in xrange(len(order)-1)]))):
                raise Exception ('Local maxima and minima not sorted correctly after collection')

def janus_collect (hist, **kwargs):
	"""
	Collect last maxima as a single (isotropic liquid) phase, and all others as a micellar gas.

	Parameters
	----------
	hist : histogram
		Histogram that has been already segmented

	"""

	# Check ln(PI) has been segmented first
	if ('ln(PI)_maxima_idx' not in hist.data): 
		raise Exception('Histogram has not been segmented yet')
	if ('ln(PI)_minima_idx' not in hist.data):
		raise Exception('Histogram has not been segmented yet')
	
	# Also double check for correct order
	check_order_ (hist)

	# Reassigned maxima and minima indices as desired if and only if there are more than 2 peaks.
	# Otherwise, take these two peaks to correspond to each phase correctly.
	if (len(hist.data['ln(PI)_maxima_idx']) > 2):
		max_idx = [int(round(np.mean(hist.data['ln(PI)_maxima_idx'][:-1]))), hist.data['ln(PI)_maxima_idx'][-1]]
		if (hist.data['ln(PI)_minima_idx'][0] > 0):
			# (First) Maxima exists at N = 0 instead of a minima
			min_idx = []
		else:
			# (First) Minima exists at N = 0
                        min_idx = [0]

		last = hist.data['ln(PI)_minima_idx'][-1]
		if (last > max_idx[0] and last < max_idx[1]):
			# Last minima in between maxima, so end of distribution is also a maxima
			min_idx.append(last)
		elif (last > max_idx[1]):
			# Last minima occurs at the end of the distribution
			assert (len(hist.data['ln(PI)_minima_idx']) > 1)
			min_idx.append(hist.data['ln(PI)_minima_idx'][-2])
			min_idx.append(hist.data['ln(PI)_minima_idx'][-1])
		
	# Check that new order alternates correctly
	check_order_ (hist)

	# Assign to the histogam
	hist.data['ln(PI)_maxima_idx'] = max_idx
	hist.data['ln(PI)_minima_idx'] = min_idx
