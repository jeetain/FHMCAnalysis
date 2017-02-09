"""
@author Nathan A. Mahynski
@date 02/09/2017
@filename moments_histogram_one_dim_gc_ntot.py
@brief Tests for isopleths generated from using one_dim_gc_ntot in histogram module
"""

import unittest, sys
sys.path.append('../../')
import FHMCAnalysis, copy, os
import numpy as np
import FHMCAnalysis.moments.histogram.one_dim.ntot.gc_binary as gcB

class TestHistogram(unittest.TestCase):
	"""
	Test histogram setup, calculations, and properties
	"""

	def setUp(self):
		"""
		Set up the class
		"""

		self.tol = 1.0e-9

	def test_combine_isopleth_grids_fail(self):
		"""
		Test it grid combination catches bad input
		"""

		mu1 = np.linspace(-15, -10, 10)
		dmu2 = np.linspace(-5, -3, 5)
		x1, y1 = np.meshgrid(mu1, dmu2)
		z1 = (x1**2+y1**2)

		mu1 = np.linspace(-10, -5, 10)
		dmu2 = np.linspace(-5, -4, 5)
		x2, y2 = np.meshgrid(mu1, dmu2)
		z2 = (x2**2+y2**2)

		# Misaligned dmu2
		fail = False
		try:
			gcB.combine_isopleth_grids([x2,x1], [y2,y1], [z2,z1])
		except:
			fail = True
		self.assertTrue(fail)

		mu1 = np.linspace(-10, -5, 10)
		dmu2 = np.linspace(-5, -3, 6)
		x2, y2 = np.meshgrid(mu1, dmu2)
		z2 = (x2**2+y2**2)

		# Unequal dmu2
		fail = False
		try:
			gcB.combine_isopleth_grids([x2,x1], [y2,y1], [z2,z1])
		except:
			fail = True
		self.assertTrue(fail)


	def test_combine_isopleth_grids_pass(self):
		"""
		Test it grid combination works correctly when good input given
		"""

		mu1 = np.linspace(-15, -10, 10)
		dmu2 = np.linspace(-5, -3, 5)
		x1, y1 = np.meshgrid(mu1, dmu2)
		z1 = (x1**2+y1**2)

		mu1 = np.linspace(-10, -5, 10)
		dmu2 = np.linspace(-5, -3, 5)
		x2, y2 = np.meshgrid(mu1, dmu2)
		z2 = (x2**2+y2**2)

		mu1 = np.concatenate((np.linspace(-15, -10, 10), np.linspace(-10, -5, 10)[1:]), axis=0)
		x3, y3 = np.meshgrid(mu1, dmu2)
		z3 = (x3**2+y3**2)

		Z, (X, Y) = gcB.combine_isopleth_grids([x2,x1], [y2,y1], [z2,z1])

		self.assertTrue(np.all(np.abs(X-x3) < self.tol))
		self.assertTrue(np.all(np.abs(Y-y3) < self.tol))
		self.assertTrue(np.all(np.abs(Z-z3) < self.tol))

if __name__ == '__main__':
	unittest.main()

	"""

	* To Do:

	* Notes:


	"""
