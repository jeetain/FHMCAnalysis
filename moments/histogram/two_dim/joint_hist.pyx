"""@docstring
@brief Create a general joint two-D histogram
@author Nathan A. Mahynski									
@date 08/06/2016									
@filename joint_hist.pyx									
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

class joint_hist (object):	
	"""
	Joint histogram
	
	"""

	class entry (object):
		"""
		Entry in joint histogram, i.e., a specific lnPI(op2) vector.

		For instance, lnPI(N).

		"""

		def __init__(self):
			"""
			Instantiate class
		
			"""

			self.clear_all()
		
		def clear_all(self):
			"""
			Clear all data in entry.

			"""

			self.data = {}
		
		def clear_props(self):
			"""
			Clear just the properties stored in an entry.

			"""

			self.data['props'] = {}
		
		def set(self, lnpi, op_vals, name_val_dict):
			"""
			Set ln(PI) and properties all at once.

			Parameters
			----------
			lnpi : ndarray
				ln(PI)
			op_vals : ndarray
				Order parameter values corresponding to ln(PI)
			name_val_dict : dict
				Dictionary of {property_name: ndarray of property}

			"""	

			self.set_lnPI(lnpi, op_vals)
			for p in name_val_dict:
				self.set_prop(p, name_val_dict[p])
					
		def set_lnPI(self, lnpi, op_vals):
			"""
			Set ln(PI) in entry.

			Parameters
			----------
			lnpi : ndarray
				ln(PI)
			op_vals : ndarray
				Order parameter values corresponding to ln(PI).  Must be sorted from low to high.

			"""	

			assert(len(op_vals) == len(lnpi)), 'Size mismatch between ln(PI) and order parameters'
			self.data['ln(PI)'] = np.array(lnpi, dtype=np.float64)
			assert (np.all(sorted(op_vals) == op_vals)), 'Order parameter values are not sorted'
			self.data['op_vals'] = np.array(op_vals, dtype=np.float64)
			if ('props' in self.data):
				for x in self.data['props']:
					assert (self._check_size(self.data['props'][x])), 'Size of existing properties vectors is different from new ln(PI)'
			
		def set_prop(self, name, val):
			"""
			Set/add a property, e.g., N1 or U, to the entry.

			Parameters
			----------
			name : str 
				Name of property
			val : ndarray 
				Property itself
	
			"""	

			assert (self._check_size(val)), 'Size of new property vector is different from existing ones'
			if ('props' not in self.data):
				self.data['props'] = {}
			self.data['props'][name] = val

		def _check_size(self, x):
			"""
			Check that the size of x is the same as properties inside the entry.
			
			Parameters
			----------
			x : ndarray 
				New property

			"""	

			cdef int ref_size = 0
			if ('ln(PI)' in self.data):
				ref_size = len(self.data['ln(PI)'])
			elif ('op_vals' in self.data):
				ref_size = len(self.data['op_vals'])
			elif ('props' in self.data):
				if (len(self.data['props']) > 0):
					dummy = [y for y in self.data['props']]
					ref_size = len(self.data['props'][dummy[0]])
				else:
					ref_size = len(x)
			else:
				ref_size = len(x)
			
			return len(x) == ref_size
			
	def __init__(self):
		"""
		Create a joint probability distribution histogram from individual ones.

		Result is lnPI(op1, op2), e.g., lnPI(h, N) surface 

		"""

		self.clear()
					
	def clear(self):
		"""
		Clear all data in this joint histogram.

		"""

		self.data = {}
	
	def add(self, op1, entry):
		"""
		Add an entry to this joint histogram.

		Parameters
		----------
		op1 : double
			Number value of order parameter 1 this entry corresponds to, e.g., h
		entry : entry 
			entry to add to joint histogram

		"""

		if ('entries' not in self.data):
			self.data['entries'] = {}
		self.data['entries'][op1] = copy.deepcopy(entry)

	def enter(self, op1, lnpi, op_vals, name_val_dict):
		"""
		Add an entry to this joint histogram by providing raw data for entry.

		Parameters
		----------
		op1 : double
			Number value of order parameter 1 this entry corresponds to, e.g., h
		lnpi : ndarray 
			ln(PI)
		op_vals : ndarray
			Order parameter values corresponding to ln(PI), e.g., ntot
		name_val_dict : dict
			Dictionary of {property_name: ndarray of property}

		"""

		e = self.entry()
		e.set(lnpi, op_vals, name_val_dict)
		self.add(op1, e)
	
	def make(self):
		"""
		Take all raw entries and sort to create a self-consistent joint probability surface.
		
		"""

		cdef int i, j, y
	
		op1_vals = sorted(self.data['entries'])
		op2_vals = []
		for x in op1_vals:
			# find unique values and sort
			op2_vals = sorted(set(op2_vals)|set(self.data['entries'][x].data['op_vals']))

		self.data['ln(PI)'] = np.full((len(op1_vals), len(op2_vals)), -np.inf, dtype=np.float64)
		self.data['op_1'] = np.array(op1_vals, dtype=np.float64)
		self.data['op_2'] = np.array(op2_vals, dtype=np.float64)
		self.data['bounds_idx'] = np.full((len(op1_vals), 2), 0, dtype=np.int)
		self.data['props'] = {}

		all_props = []
		for j in xrange(len(op1_vals)):
			x = op1_vals[j]
			op2 = self.data['entries'][x].data['op_vals']
			lnpi = self.data['entries'][x].data['ln(PI)']
			props = sorted([p for p in self.data['entries'][x].data['props']])
			min_idx = np.inf
			max_idx = -np.inf
			for i in xrange(len(op2)):
				y = op2_vals.index(op2[i])
				self.data['ln(PI)'][j,y] = lnpi[i]
				min_idx = np.min([min_idx, y])
				max_idx = np.max([max_idx, y])
			self.data['bounds_idx'][j,:] = [min_idx, max_idx]
			if (len(all_props) > 0):
				assert (props == all_props), 'Properties are not all the same, or some are missing'
			else:
				all_props = copy.copy(props)
			
		for prop in all_props:
			self.data['props'][prop] = np.full((len(op1_vals), len(op2_vals)), 0, dtype=np.float64)
			for j in xrange(len(op1_vals)):
				x = op1_vals[j]
				op2 = self.data['entries'][x].data['op_vals']
				for i in xrange(len(op2)):
					y = op2_vals.index(op2[i])
					self.data['props'][prop][j,y] = self.data['entries'][x].data['props'][prop][i]
	
	def to_json(self, fname):
		"""
		Print this joint histogram's data to a json file.

		Parameters
		----------
		fname : str
			Filename to print to

		"""

		obj = copy.deepcopy(self.data)
		obj.pop('entries', None)
		obj['ln(PI)'] = obj['ln(PI)'].tolist()
		obj['op_1'] = obj['op_1'].tolist()
		obj['op_2'] = obj['op_2'].tolist()
		obj['bounds_idx'] = obj['bounds_idx'].tolist()
		for p in obj['props']:
			obj['props'][p] = obj['props'][p].tolist()
		f = open(fname, 'w')
		json.dump(obj, f, indent=4, sort_keys=True)
		f.close()
	
	def from_json(self, fname):
		"""
		Read data from json file.

		Parameters
		----------
		fname : str
			Filename to read from

		"""

		self.clear()
		f = open(fname, 'r')
		raw = json.load(f)
		f.close()
		
		assert ('ln(PI)' in raw), 'Missing ln(PI) information'
		assert ('op_1' in raw), 'Missing op_1 information'
		assert ('op_2' in raw), 'Missing op_2 information'
		assert ('bounds_idx' in raw), 'Missing bounds information'
		assert ('props' in raw), 'Missing properties information'
		
		self.data['ln(PI)'] = np.array(raw['ln(PI)'], dtype=np.float64)
		self.data['op_1'] = np.array(raw['op_1'], dtype=np.float64)
		self.data['op_2'] = np.array(raw['op_2'], dtype=np.float64)
		self.data['bounds_idx'] = np.array(raw['bounds_idx'], dtype=np.float64)
		
		self.data['props'] = {}
		for p in raw['props']:
			self.data['props'][p] = np.array(raw['props'][p], dtype=np.float64)
		
if __name__ == '__main__':
	print 'joint_hist.pyx'
	
	"""
	
	* Tutorial:
	
	1.) Instantiate the joint_hist
	2.) Use the add() member to add data to the histogram (in any order), or use entry()
	3.) Use make() to order and produce the joint histogram for further use/manipulation
	4.) Print to file or use this class to perform other operations

	* Notes:
	
	Histogram is produced which guarantees that the order parameters are sorted from low to high in both dimensions.
	However, no guarantee is placed on the value of these bounds.

	* To Do:

	"""
