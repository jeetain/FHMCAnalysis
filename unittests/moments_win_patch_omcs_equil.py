"""
@author Nathan A. Mahynski
@date 07/19/2016
@filename moments_win_patch_equil.py
@brief Tests for omcs_equil in win_patch module
"""

import unittest, sys
sys.path.append('../../')
import FHMCAnalysis, copy, os
import numpy as np
import FHMCAnalysis.moments.win_patch.omcs_patch as wP
import FHMCAnalysis.moments.win_patch.omcs_equil as eQ

class TestWindowEquil(unittest.TestCase):
	"""
	Test window equilibration estimation
	"""

	def setUp(self):
		"""
		Set up class
		"""

		self.source = 'reference/test_sim/'
	
	def test_find_windows(self):
		"""
		Check that we can get a sequence of windows from a file tree of varying complexity
		"""	
	
		fail = False
		try:
			windows, nebr_set = eQ.find_windows(self.source)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)	
		
		self.assertTrue(np.all(windows == [1,2]))
		self.assertTrue(np.all(nebr_set == [(1,2)]))
	
	def test_nebr_win_conv(self):
		"""
		Check neighbor window convergence test
		"""	
	
		match, uerr, nerr = eQ.test_window_match(self.source+'1/', self.source+'2/', 1.0)
		self.assertTrue(not match) # too stringent of convergence criterion
		match, uerr, nerr = eQ.test_window_match(self.source+'1/', self.source+'2/', 10.0)
		self.assertTrue(match) # less stringent of convergence criterion
		self.assertTrue(np.abs(uerr - 4.31410893236) < 1.0e-8)
		self.assertTrue(np.abs(nerr - 8.04638999443) < 1.0e-8)
	
	def test_nebr_seq_conv(self):
		"""
		Check neighbor sequence convergence test
		"""	
	
		seq = wP.get_patch_sequence(self.source)
		match, uerr, nerr = eQ.test_nebr_match(seq[0], seq[1], 1.0)
		self.assertTrue(not match) # too stringent of convergence criterion
		match, uerr, nerr = eQ.test_nebr_match(seq[0], seq[1], 10.0)
		self.assertTrue(match) # less stringent of convergence criterion
		self.assertTrue(np.abs(uerr - 4.31410893236) < 1.0e-8)
		self.assertTrue(np.abs(nerr - 8.04638999443) < 1.0e-8)	
	
	def test_nebr_equil(self):
		"""
		Check neighbor convergence workflow
		"""	
	
		seq = wP.get_patch_sequence(self.source)
		
		# too conservative, none converged
		fail = False
		try:
			refined_seq = eQ.test_nebr_equil(seq, 1.0, 'maxEq', False)
		except Exception as e:
			fail = True
		self.assertTrue(fail)
		
		# found counverged, but don't trust last one
		fail = False
		try:
			refined_seq = eQ.test_nebr_equil(seq, 10.0, 'maxEq', False)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(len(refined_seq) == 1)
		
		# found converged, and trust last one
		fail = False
		try:
			refined_seq = eQ.test_nebr_equil(seq, 10.0, 'maxEq', True)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(len(refined_seq) == 2)
		
		if ('maxEq' in os.listdir('./')):
			os.remove('maxEq')
		
if __name__ == '__main__':
    unittest.main()
