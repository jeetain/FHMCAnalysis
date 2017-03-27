"""
@author Nathan A. Mahynski
@date 07/19/2016
@filename moments_win_patch_fhmc.py
@brief Tests for fhmc_patch in win_patch module
"""

import unittest, sys
sys.path.append('../../')
import FHMCAnalysis, copy, os
import numpy as np
import FHMCAnalysis.moments.win_patch.fhmc_patch as wP

class TestLocalNHist(unittest.TestCase):
	"""
	Test local N histogram
	"""

	def setUp(self):
		"""
		Set up the class
		"""

		self.fname = 'reference/test_pk_hist.dat'
		self.fname2 = 'reference/test_pk_hist2.dat'
		self.fname3 = 'reference/test_pk_hist3.dat'

	def test_load(self):
		"""
		Test it loads correctly
		"""

		fail = False
		try:
			lh = wP.local_hist(self.fname)
		except:
			fail = True
		self.assertTrue(not fail)

	def test_clear(self):
		"""
		Test it clears correctly
		"""

		lh = wP.local_hist(self.fname)
		lh.clear()
		self.assertEqual(len(lh.lb), 0)
		self.assertEqual(len(lh.ub), 0)
		self.assertEqual(len(lh.bw), 0)
		self.assertEqual(len(lh.h), 0)
		self.assertEqual(lh.win_start, 0)
		self.assertEqual(lh.win_end, 0)

	def test_merge_ov_lower(self):
		"""
		Merge two histograms and overwrite the lower one (other)
		"""

		lh1 = wP.local_hist(self.fname)
		lh2 = wP.local_hist(self.fname2)
		fail = False
		try:
			lh2.merge(lh1, 0.0)
		except:
			fail = True
		self.assertTrue(not fail)

		for i in xrange(0, 19):
			self.assertTrue (len(lh2.h[i]) == len(lh1.h[i]))
			for j in xrange(len(lh2.h[i])):
				self.assertTrue (lh2.h[i][j] == lh1.h[i][j])

		self.assertTrue (np.all(lh2.h[19] == [0.91,0.08,0.005,0.005]))
		self.assertTrue (np.all(lh2.h[20] == [0.9,0.09,0.005,0.005,0.]))
		self.assertTrue (np.all(lh2.h[21] == [0.4,0.3,0.2,0.1]))
		self.assertTrue (np.all(lh2.h[22] == [0.05,0.05,0.2,0.3,0.4]))

	def test_merge_ov_upper(self):
		"""
		Merge two histograms and overwrite the upper one (self)
		"""

		lh1 = wP.local_hist(self.fname)
		lh2 = wP.local_hist(self.fname2)
		fail = False
		try:
			lh2.merge(lh1, 1.0)
		except:
			fail = True
		self.assertTrue(not fail)

		for i in xrange(0, 21):
			self.assertTrue (len(lh2.h[i]) == len(lh1.h[i]))
			for j in xrange(len(lh2.h[i])):
				self.assertTrue (lh2.h[i][j] == lh1.h[i][j])

		self.assertTrue (np.all(lh2.h[21] == [0.4,0.3,0.2,0.1]))
		self.assertTrue (np.all(lh2.h[22] == [0.05,0.05,0.2,0.3,0.4]))

	def test_merge_ave(self):
		"""
		Merge two histograms and average the two
		"""

		lh1 = wP.local_hist(self.fname)
		lh2 = wP.local_hist(self.fname2)
		fail = False
		try:
			lh2.merge(lh1, 0.5)
		except:
			fail = True
		self.assertTrue(not fail)

		for i in xrange(0, 19):
			self.assertTrue (len(lh2.h[i]) == len(lh1.h[i]))
			for j in xrange(len(lh2.h[i])):
				self.assertTrue (lh2.h[i][j] == lh1.h[i][j])

		self.assertTrue (np.all(abs(lh2.h[19] - [0.89158012,0.09900905,0.006652,0.00275883]) < 1.0e-6))
		self.assertTrue (np.all(abs(lh2.h[20] - [8.90009879e-01,1.01615354e-01,5.73284601e-03,2.63270520e-03,9.21563857e-06]) < 1.0e-6))
		self.assertTrue (np.all(lh2.h[21] == [0.4,0.3,0.2,0.1]))
		self.assertTrue (np.all(lh2.h[22] == [0.05,0.05,0.2,0.3,0.4]))

	def test_merge_ave2(self):
		"""
		Merge two histograms and average the two for another case
		"""

		lh1 = wP.local_hist(self.fname)
		lh2 = wP.local_hist(self.fname3)
		fail = False
		try:
			lh2.merge(lh1, 0.5)
		except:
			fail = True
		self.assertTrue(not fail)

		for i in xrange(0, 19):
			self.assertTrue (len(lh2.h[i]) == len(lh1.h[i]))
			for j in xrange(len(lh2.h[i])):
				self.assertTrue (lh2.h[i][j] == lh1.h[i][j])

		self.assertTrue (np.all(abs(lh2.h[19] - [0.89158012,0.09900905,0.006652,0.00275883]) < 1.0e-6))
		self.assertTrue (np.all(abs(lh2.h[20] - [0.44000988,0.50661535,0.04823285,0.00263271,0.00250922]) < 1.0e-6))
		self.assertTrue (np.all(lh2.h[21] == [0.4,0.3,0.2,0.1]))
		self.assertTrue (np.all(lh2.h[22] == [0.05,0.05,0.2,0.3,0.4]))

	def test_merge_renorm(self):
		"""
		Merge two histograms and average the two, then renormalizes each histogram
		"""

		lh1 = wP.local_hist(self.fname)
		lh2 = wP.local_hist(self.fname2)
		fail = False
		try:
			lh2.merge(lh1, 0.5)
		except:
			fail = True
		self.assertTrue(not fail)

		# artificially multiply by 2
		for l in lh2.h:
			x = np.array(l)*2
			l = x.tolist()
			self.assertTrue (abs(np.sum(l) - 2.0) < 1.0e-8)

		fail = False
		try:
			lh2.normalize()
		except:
			fail = True
		self.assertTrue(not fail)

		# check renormalized to 1
		for l in lh2.h:
			self.assertTrue(abs((np.sum(l) - 1.0) < 1.0e-8))

class TestLocalEHist(unittest.TestCase):
	"""
	Test local E histogram
	"""

	def setUp(self):
		self.fname = 'reference/test_e_hist.dat'
		self.fname2 = 'reference/test_e_hist2.dat'

	def test_load(self):
		"""
		Test it loads correctly
		"""

		fail = False
		try:
			lh = wP.local_hist(self.fname)
		except:
			fail = True
		self.assertTrue(not fail)

	def test_clear(self):
		"""
		Test it clears correctly
		"""

		lh = wP.local_hist(self.fname)
		lh.clear()
		self.assertEqual(len(lh.lb), 0)
		self.assertEqual(len(lh.ub), 0)
		self.assertEqual(len(lh.bw), 0)
		self.assertEqual(len(lh.h), 0)
		self.assertEqual(lh.win_start, 0)
		self.assertEqual(lh.win_end, 0)

	def test_merge_ov_lower(self):
		"""
		Merge two histograms and overwrite the lower one (other)
		"""

		lh1 = wP.local_hist(self.fname)
		lh2 = wP.local_hist(self.fname2)
		fail = False
		try:
			lh2.merge(lh1, 0.0)
		except:
			fail = True
		self.assertTrue(not fail)

		for i in xrange(0, 20):
			self.assertTrue (len(lh2.h[i]) == len(lh1.h[i]))
			for j in xrange(len(lh2.h[i])):
				self.assertTrue (lh2.h[i][j] == lh1.h[i][j])

		self.assertTrue (lh2.h[20][1] == 1)
		self.assertTrue (np.all(np.abs([lh2.h[20][x] for x in xrange(len(lh2.h[20])) if x!= 1]) < 1.0e-8))
		self.assertTrue (np.all(lh2.h[21] == [0.1,0.1,0.1,0.4,0.3]))

	def test_merge_ov_upper(self):
		"""
		Merge two histograms and overwrite the upper one (self)
		"""

		lh1 = wP.local_hist(self.fname)
		lh2 = wP.local_hist(self.fname2)
		fail = False
		try:
			lh2.merge(lh1, 1.0)
		except:
			fail = True
		self.assertTrue(not fail)

		for i in xrange(0, 21):
			self.assertTrue (len(lh2.h[i]) == len(lh1.h[i]))
			for j in xrange(len(lh2.h[i])):
				self.assertTrue (lh2.h[i][j] == lh1.h[i][j])

		self.assertTrue (np.all(lh2.h[21] == [0.1,0.1,0.1,0.4,0.3]))

	def test_merge_ave(self):
		"""
		Merge two histograms and average the two
		"""

		lh1 = wP.local_hist(self.fname)
		lh2 = wP.local_hist(self.fname2)
		fail = False
		try:
			lh2.merge(lh1, 0.5)
		except:
			fail = True
		self.assertTrue(not fail)

		for i in xrange(0, 20):
			self.assertTrue (len(lh2.h[i]) == len(lh1.h[i]))
			for j in xrange(len(lh2.h[i])):
				self.assertTrue (lh2.h[i][j] == lh1.h[i][j])

		self.assertTrue (np.abs(lh2.h[20][1] - (1+0.00105795530783919)/2.0) < 1.0e-8)
		a = np.array([lh2.h[20][x] for x in xrange(len(lh2.h[20])) if x!= 1])
		b = np.array([lh1.h[20][x] for x in xrange(len(lh1.h[20])) if x!= 1])/2.0
		self.assertTrue (np.all(np.abs(a-b) < 1.0e-8))
		self.assertTrue (np.all(lh2.h[21] == [0.1,0.1,0.1,0.4,0.3]))

class TestWindow(unittest.TestCase):
	"""
	Test window behavior
	"""

	def setUp(self):
		self.source = 'reference/test_sim/'

	def test_get_seq(self):
		"""
		Check that we can get a sequence from a source tree of varying complexity
		"""

		fail = False
		try:
			seq = wP.get_patch_sequence(self.source)
		except Exception as e:
			print e
			fail = True
		self.assertTrue (not fail)

	def test_init(self):
		"""
		Initialize a window for patching
		"""

		seq = wP.get_patch_sequence(self.source)
		fail = False
		try:
			wh = wP.window (seq[0][0], seq[0][1], seq[0][2], seq[0][3], 2, False)
		except Exception as e:
			fail = True
			print e
		self.assertTrue(not fail)

	def test_repr(self):
		"""
		Check self-representation
		"""

		seq = wP.get_patch_sequence(self.source)
		wh = wP.window (seq[0][0], seq[0][1], seq[0][2], seq[0][3], 2, False)
		a = seq[0][0]+"::"+seq[0][1]+"::"+seq[0][2]+"::"+seq[0][3]+"-[0,20]"
		self.assertTrue(a == str(wh))

	def test_clear(self):
		"""
		Check ability to clear all data from window
		"""

		seq = wP.get_patch_sequence(self.source)
		wh = wP.window (seq[0][0], seq[0][1], seq[0][2], seq[0][3], 2, False)
		self.assertTrue (len(wh.lnPI) != 0)
		self.assertTrue (wh.nspec != 0)
		wh.clear()
		self.assertTrue (len(wh.lnPI) == 0)
		self.assertTrue (wh.nspec == 0)

	def test_load_info(self):
		"""
		Check information loaded correctly at init
		"""

		seq = wP.get_patch_sequence(self.source)
		wh = wP.window (seq[0][0], seq[0][1], seq[0][2], seq[0][3], 2, False)
		a = np.array([0.0, 11.5792872, 22.55514816, 33.16632265, 43.53878289, 53.80927566, 63.94826804, 73.97895064, 83.96576198, 93.94840544, 103.8773032, 113.77306514, 123.71227577, 133.68404802, 143.69837309, 153.86625598, 164.18813354, 174.70527468, 185.3787788, 196.24217909, 207.27150728])
		self.assertTrue(len(wh.lnPI) == 21)
		self.assertTrue(np.all(np.abs(a - wh.lnPI) < 1.0e-6))

		self.assertTrue(wh.mom.shape == (36*3,21))

		# check symmetry of moments
		for i in xrange(0, 2):
			for j in xrange(0, 3):
				for k in xrange(0, 2):
					for m in xrange(0, 3):
						for p in xrange(0, 3):
							# N_i^j*N_k^m*U^0 == N_k^m*N_i^j*U^0
							address1 = p + 3*m + 3*3*k + 3*3*2*j + 3*3*2*3*i
							address2 = p + 3*j + 3*3*i + 3*3*2*m + 3*3*2*3*k
							self.assertTrue(np.all(wh.mom[address1] == wh.mom[address2]))

							if (j == m and m == 0 and p == 0):
								# all raised to 0 power so should be 1
								self.assertTrue(np.all(np.abs(wh.mom[address1] - np.ones(len(wh.mom[address1]))) < 1.0e-8))

							if (i == k and m + j < 3):
								# e.g., N_1*N_1*U^p = N_1^2*N_1^0*U^p, also true that N_1*N_1*U^p = N_1^2*N_x^0*U^p
								for kk in xrange(0, 2):
									address2 = p + 3*0 + 3*3*kk + 3*3*2*(j+m) + 3*3*2*3*i
									self.assertTrue(np.all(wh.mom[address1] == wh.mom[address2]))

		# check some histogram information from pk and energy
		x = [0,0,-1,-3,-6,-10,-11,-15,-20,-26,-32,-41,-43,-49,-55,-67,-73,-82,-88,-94,-102]
		self.assertTrue(np.all(wh.e_hist.lb == x))
		x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-3,-4,-7,-11]
		self.assertTrue(np.all(wh.e_hist.ub == x))
		x = np.ones(21)
		self.assertTrue(np.all(wh.e_hist.bw == x))
		for i in xrange(len(wh.e_hist.h)):
			self.assertTrue(len(wh.e_hist.h[i]) == wh.e_hist.ub[i]-wh.e_hist.lb[i]+1)
		self.assertTrue(np.all(np.abs(wh.e_hist.h[3] - np.array([0.00907625393757033,0.0185828627062264,0.248847389827399,0.723493493528804])) < 1.0e-8))
		self.assertTrue(len(wh.pk_hist) == 2)

	def test_merge_no_smooth_lnpi(self):
		"""
		Check lnPI merging properly without smoothing
		"""
		seq = wP.get_patch_sequence(self.source)
		wh1 = wP.window (seq[0][0], seq[0][1], seq[0][2], seq[0][3], 1, False)
		wh2 = wP.window (seq[1][0], seq[1][1], seq[1][2], seq[1][3], 1, False)
		ref_lnpi = copy.deepcopy(wh2.lnPI)

		fail = False
		try:
			shift, e2 = wh2.merge(wh1)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)

		# use lower lnPI outside overlap
		self.assertTrue (np.all(np.abs(wh2.lnPI[:17] - wh1.lnPI[:17]) < 1.0e-6))
		# in overlapping region (less offset = 1), just use lower lnPI
		self.assertTrue (np.all(np.abs(wh2.lnPI[16+1:21-1] - wh1.lnPI[16+1:21-1]) < 1.0e-6))
		# above the overlap, just use larger lnPI
		self.assertTrue (np.all(np.abs(wh2.lnPI[20:] - (ref_lnpi[4:]+shift)) < 1.0e-6))

	def test_merge_no_smooth_mom(self):
		"""
		Check moments merging properly without smoothing
		"""

		seq = wP.get_patch_sequence(self.source)
		wh1 = wP.window (seq[0][0], seq[0][1], seq[0][2], seq[0][3], 1, False)
		wh2 = wP.window (seq[1][0], seq[1][1], seq[1][2], seq[1][3], 1, False)
		ref_mom = copy.deepcopy(wh2.mom)

		fail = False
		try:
			shift, e2 = wh2.merge(wh1)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)

		self.assertTrue(wh2.mom.shape == (36*3,31))

		# use lower nn_mom outside overlap
		self.assertTrue(np.all(np.abs(wh2.mom[:,:17] - wh1.mom[:,:17]) < 1.0e-6))
		# in overlapping region (less offset = 1), just use lower nn_mom
		self.assertTrue(np.all(np.abs(wh2.mom[:,16+1:21-1] - wh1.mom[:,16+1:21-1]) < 1.0e-6))
		# above the overlap, just use larger nn_mom
		self.assertTrue(np.all(np.abs(wh2.mom[:,20:] - ref_mom[:,4:]) < 1.0e-6))

	def test_merge_with_smooth_lnpi(self):
		"""
		Check lnPI merging properly with smoothing
		"""

		seq = wP.get_patch_sequence(self.source)
		wh1 = wP.window (seq[0][0], seq[0][1], seq[0][2], seq[0][3], 1, True)
		wh2 = wP.window (seq[1][0], seq[1][1], seq[1][2], seq[1][3], 1, True)
		ref_lnpi = copy.deepcopy(wh2.lnPI)

		fail = False
		try:
			shift, e2 = wh2.merge(wh1)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)

		# use lower lnPI outside overlap
		self.assertTrue (np.all(np.abs(wh2.lnPI[:17] - wh1.lnPI[:17]) < 1.0e-6))
		# in overlapping region (less offset = 1), average the lnPI, check that answer is close to average
		self.assertTrue (np.all(np.abs((wh2.lnPI[16+1:21-1]-wh1.lnPI[16+1:21-1])/(0.5*(wh2.lnPI[16+1:21-1]+wh1.lnPI[16+1:21-1]))) < 1.0e-3))
		# above the overlap, just use larger lnPI
		self.assertTrue (np.all(np.abs(wh2.lnPI[20:] - (ref_lnpi[4:]+shift)) < 1.0e-6))

	def test_merge_with_smooth_mom(self):
		"""
		Check nn moments merging properly with smoothing
		"""

		seq = wP.get_patch_sequence(self.source)
		wh1 = wP.window (seq[0][0], seq[0][1], seq[0][2], seq[0][3], 1, True)
		wh2 = wP.window (seq[1][0], seq[1][1], seq[1][2], seq[1][3], 1, True)
		ref_mom = copy.deepcopy(wh2.mom)

		fail = False
		try:
			shift, e2 = wh2.merge(wh1)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)

		# use lower nn_mom outside overlap
		self.assertTrue(np.all(np.abs(wh2.mom[:,:17] - wh1.mom[:,:17]) < 1.0e-6))
		# in overlapping region (less offset = 1), average the nn_mom, check that average deviation from the average is "small" (depends on data quality)
		self.assertTrue(np.average(np.abs(wh2.mom[:,16+1:21-1]-wh1.mom[:,16+1:21-1])/(0.5*(wh2.mom[:,16+1:21-1]+wh1.mom[:,16+1:21-1]))) < 0.02)
		# above the overlap, just use larger nn_mom
		self.assertTrue(np.all(np.abs(wh2.mom[:,20:] - ref_mom[:,4:]) < 1.0e-6))

	def test_to_nc(self):
		"""
		Check print to netCDF4 function
		"""

		seq = wP.get_patch_sequence(self.source)
		wh1 = wP.window (seq[0][0], seq[0][1], seq[0][2], seq[0][3], 1, True)
		wh2 = wP.window (seq[1][0], seq[1][1], seq[1][2], seq[1][3], 1, True)

		fail = False
		try:
			shift, e2 = wh2.merge(wh1)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)

		try:
			wh2.to_nc("test.nc")
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)

		fail = False
		if ('test.nc' in os.listdir('./')):
			os.remove('test.nc')
		else:
			fail = True
		self.assertTrue(not fail)

	def test_patch_all(self):
		"""
		Test complete window patching of a directory
		"""

		fail = False
		try:
			seq = wP.get_patch_sequence(self.source)
			max_err_name, max_err_val = wP.patch_all_windows (seq, 'composite.nc', 'patch.log', 1, False)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)

		fail = False
		if ('composite.nc' in os.listdir('./')):
			os.remove('composite.nc')
		else:
			fail = True
		self.assertTrue(not fail)

		fail = False
		if ('patch.log' in os.listdir('./')):
			os.remove('patch.log')
		else:
			fail = True
		self.assertTrue(not fail)

if __name__ == '__main__':
    unittest.main()
