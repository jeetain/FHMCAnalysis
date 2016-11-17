"""@docstring
@brief F(h) profiles for pores
@author Nathan A. Mahynski									
@date 09/01/2016									
@filename free_energy_profile.pyx									
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
from numpy.polynomial.polynomial import polyval
from scipy import interpolate

class interp (object):
	"""
	Class that takes a file with points in column format: (h, F(h)) and interpolates (linearly) to find F(h)	at arbitrary h. 

	If asked to interpolate outside of bounds, the maximum value of F(h) given is returned. If unable to load text from a file, will force quit.
	
	"""
	
	def __init__ (self, filename):
		"""
		Instatiate the clas

		Parameters
		----------
		filename : str
			File to read (h, F(h)) from	

		"""

		self.filename = filename
		try:
			raw = np.loadtxt(self.filename, comments="#")
			self.h = np.array([i[0] for i in raw])
			self.f = np.array([i[1] for i in raw])
		except:
			print "Unable to read profile from "+str(self.filename)
			sys.exit() # Force quit
		self.interpolate = interpolate.interp1d(self.h, self.f, bounds_error=False, fill_value=np.max(self.f))
        
	def free_energy (self, double h):
		"""
		Returns the interpolated free energy at a given coordinate, h
	
		Parameters
		----------
		h : double
			Coordinate to interpolate F(h) at

		Returns
		-------
		double
			F(h)

		"""

		return self.interpolate(h)

class polynomial (object):
	"""
	Polynomial class that, given coefficients in leading order, will return F(h) as a polynomial of 	that order.

	"""

	def __init__ (self, C): # Specify from leading order, e.g. C[0]*x^n + C[1]*x^(n-1) + ... + C[n]
		"""
		Instatiate the class

		Parameters
		----------
		C : array
			Coefficients, C[0]*x^n + C[1]*x^(n-1) + ... + C[n]

		"""

		self.coeffs = C[::-1]
		self.order = len(self.coeffs) - 1

	def free_energy (self, h):
		"""
		Calculate the free energy

		Parameters
		----------
		h : double
			Value of h to compute F(h) at

		Returns
		-------
		double
			F(h)

		"""

		return polyval(h,self.coeffs)

if __name__ == '__main__':
	print 'free_energy_profile.pyx'
	
	"""
	
	* Tutorial:
	
	* Notes:
	
	* To Do:

	"""
