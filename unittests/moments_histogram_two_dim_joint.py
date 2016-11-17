"""
@author Nathan A. Mahynski
@date 07/19/2016
@filename moments_histogram_two_dim_joint.py
@brief Tests for two dimensional joint histogram class
"""

import unittest, sys
sys.path.append('../../')
import FHMCAnalysis, copy, os
import numpy as np
import FHMCAnalysis.moments.histogram.two_dim.joint_hist as jH

class test_joint_hist(unittest.TestCase):
	def setUp(self):
		self.hist = jH.joint_hist()
		self.fname = 'reference/joint_test.json'

	"""
	Test entry init
	"""
	def test_entry(self):
		fail = False
		try:
			e = self.hist.entry()
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)	
	
	"""
	Test entry set lnPI
	"""
	def test_entry_set_lnPI(self):
		lnpi = np.array([1,2,3])
		ntot = np.array([0,1,2])
		en = self.hist.entry()
		fail = False
		try:
			en.set_lnPI(lnpi, ntot)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		
	"""
	Test entry set properties 
	"""
	def test_entry_set_props(self):
		props = {'U':np.array([5,5,5]), 'N2':np.array([1,4,8])}
		en = self.hist.entry()
		fail = False
		try:
			for p in props:
				en.set_prop(p, props[p])
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
	
	"""
	Test entry set all
	"""
	def test_entry_set(self):
		props = {'U':np.array([5,5,5]), 'N2':np.array([1,4,8])}
		lnpi = np.array([1,2,3])
		ntot = np.array([0,1,2])
		en = self.hist.entry()
		fail = False
		try:
			en.set(lnpi, ntot, props)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
	
	"""
	Test bad set due to bad property
	"""		
	def test_bad_set_props(self):
		props = {'U':np.array([5,5,5]), 'N2':np.array([1,4])}
		lnpi = np.array([1,2,3])
		ntot = np.array([0,1,2])
		en = self.hist.entry()
		fail = False
		try:
			en.set(lnpi, ntot, props)
		except Exception as e:
			fail = True
		self.assertTrue(fail)
		
	"""
	Test bad set due to ln(PI)
	"""		
	def test_bad_set_lnPI1(self):
		props = {'U':np.array([5,5,5]), 'N2':np.array([1,4, 8])}
		lnpi = np.array([1,2])
		ntot = np.array([0,1])
		en = self.hist.entry()
		fail = False
		try:
			en.set(lnpi, ntot, props)
		except Exception as e:
			fail = True
		self.assertTrue(fail)	
		
	"""
	Test bad set due to ln(PI)
	"""		
	def test_bad_set_lnPI2(self):
		props = {'U':np.array([5,5,5]), 'N2':np.array([1,4, 8])}
		lnpi = np.array([1,2,3])
		ntot = np.array([0,1])
		en = self.hist.entry()
		fail = False
		try:
			en.set(lnpi, ntot, props)
		except Exception as e:
			fail = True
		self.assertTrue(fail)
	
	"""
	Add an entry to the histogram
	"""
	def test_enter(self):
		props = {'U':np.array([5,5,5]), 'N2':np.array([1,4,8])}
		lnpi = np.array([1,2,3])
		ntot = np.array([0,1,2])
		
		fail = False
		try:
			self.hist.enter(1, lnpi, ntot, props)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		
	"""
	Add an entry to the histogram and make surface
	"""
	def test_single_make(self):
		props = {'U':np.array([5,5,5]), 'N2':np.array([1,4,8])}
		lnpi = np.array([1,2,3])
		ntot = np.array([0,1,2])
		
		fail = False
		try:
			self.hist.enter(1, lnpi, ntot, props)
			self.hist.make()
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		
		self.assertTrue(np.all(self.hist.data['ln(PI)'] == [[1,2,3]]))
	
	"""
	Add double entries to the histogram and make surface
	"""
	def test_double_make(self):
		props = {'U':np.array([5,5,5]), 'N2':np.array([1,4,8])}
		lnpi = np.array([1,2,3])
		ntot = np.array([0,1,2])
		
		fail = False
		try:
			self.hist.enter(2, lnpi, ntot, props)
			self.hist.enter(1, lnpi*2, ntot, props)
			self.hist.make()
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		
		self.assertTrue(np.all(self.hist.data['ln(PI)'] == [[2,4,6],[1,2,3]]))		
	
	"""
	Test surface is created when entries vary in size
	"""
	def test_make_vary(self):
		props = {'U':np.array([5,5,5]), 'N2':np.array([1,4,8])}
		self.hist.enter(1, np.array([1,2,3]), np.array([0,1,2]), props)
		props = {'U':np.array([5,5,5,5]), 'N2':np.array([1,4,8,12])}
		self.hist.enter(2, np.array([1,2,3,4]), np.array([0,1,2,3]), props)
	
		fail = False
		try:
			self.hist.make()
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		
		self.assertTrue(np.all(self.hist.data['ln(PI)'] == [[1,2,3,-np.inf],[1,2,3,4]]))	
	
	"""
	Test surface is created when entries vary in size
	"""
	def test_make_vary2(self):
		props = {'U':np.array([5,5,5]), 'N2':np.array([1,4,8])}
		self.hist.enter(1, np.array([1,2,3]), np.array([1,2,3]), props)
		props = {'U':np.array([5,5,5,5,5]), 'N2':np.array([1,1,1,1,1])}
		self.hist.enter(2, np.array([0,1,2,3,4]), np.array([0,1,2,3,4]), props)
	
		fail = False
		try:
			self.hist.make()
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		 
		self.assertTrue(np.all(self.hist.data['ln(PI)'] == [[-np.inf,1,2,3,-np.inf],[0,1,2,3,4]]))	
		self.assertTrue(np.all(self.hist.data['op_1'] == [1,2]))
		self.assertTrue(np.all(self.hist.data['op_2'] == [0,1,2,3,4]))
		self.assertTrue(np.all(self.hist.data['bounds_idx'] == [[1,3],[0,4]]))	
	
	"""
	Test printing to json file
	"""
	def test_to_json(self):
		props = {'U':np.array([5,5,5]), 'N2':np.array([1,4,8])}
		self.hist.enter(1, np.array([1,2,3]), np.array([1,2,3]), props)
		props = {'U':np.array([5,5,5,5,5]), 'N2':np.array([1,1,1,1,1])}
		self.hist.enter(2, np.array([0,1,2,3,4]), np.array([0,1,2,3,4]), props)
		self.hist.make()
		
		fail = False
		try:
			self.hist.to_json(self.fname)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
	
	"""
	Test reading from json file
	"""
	def test_from_json(self):
		fail = False
		try:
			self.hist.from_json(self.fname)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		
		self.assertTrue(np.all(self.hist.data['ln(PI)'] == [[-np.inf,1,2,3,-np.inf],[0,1,2,3,4]]))	
		self.assertTrue(np.all(self.hist.data['op_1'] == [1,2]))
		self.assertTrue(np.all(self.hist.data['op_2'] == [0,1,2,3,4]))
		self.assertTrue(np.all(self.hist.data['bounds_idx'] == [[1,3],[0,4]]))	
	
if __name__ == '__main__':
    unittest.main()
