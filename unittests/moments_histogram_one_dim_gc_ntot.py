"""
@author Nathan A. Mahynski
@date 07/20/2016
@filename moments_histogram_one_dim_gc_ntot.py
@brief Tests for one_dim_gc_ntot in histogram module
"""

import unittest, sys
sys.path.append('../../')
import FHMCAnalysis, copy, os
import numpy as np
import FHMCAnalysis.moments.histogram.one_dim.ntot.gc_hist as oneDH

class TestHistogram(unittest.TestCase):
	def setUp(self):
		"""
		Set up the class 
		"""

		self.fname = 'reference/test.nc'
		self.beta_ref = 1.0
		self.mu_ref = [5, 0]
		self.smooth = 1
		
	def test_init(self):
		"""
		Test it initializes correctly
		"""

		fail = False
		try:
			hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(hist.metadata['beta_ref'] == self.beta_ref)
		self.assertTrue(np.all(hist.metadata['mu_ref'] == self.mu_ref))
		self.assertTrue(hist.metadata['smooth'] == self.smooth)
		self.assertTrue(hist.metadata['fname'] == self.fname)
		
	def test_load(self):
		"""
		Test it loads data correctly
		"""

		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		self.assertTrue(hist.data['ln(PI)'].shape == (31,))
		self.assertTrue(hist.data['max_order'] == 2)
		self.assertTrue(hist.data['volume'] == 729)
		self.assertTrue(np.all(hist.data['ntot'] == np.arange(0,31)))
		self.assertTrue(hist.data['lb'] == hist.data['ntot'][0])
		self.assertTrue(hist.data['ub'] == hist.data['ntot'][30])
		self.assertTrue(hist.data['pk_hist']['hist'].shape == (2, 31, 122))	
		self.assertTrue(hist.data['pk_hist']['lb'].shape == (2, 31))
		self.assertTrue(hist.data['pk_hist']['ub'].shape == (2, 31))
		self.assertTrue(hist.data['pk_hist']['bw'].shape == (2, 31))
		self.assertTrue(hist.data['e_hist']['hist'].shape == (31, 122))	
		self.assertTrue(hist.data['e_hist']['lb'].shape == (31,))
		self.assertTrue(hist.data['e_hist']['ub'].shape == (31,))
		self.assertTrue(hist.data['e_hist']['bw'].shape == (31,))
		self.assertTrue(hist.data['mom'].shape == (2,3,2,3,3,31))
	
	def testClear(self):
		"""
		Test data is cleared
		"""
	
		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		fail = False
		try:
			hist.clear()
		except:
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(len(hist.data) == 0)
		self.assertTrue(len(hist.metadata) != 0)
	
	def testNorm(self):
		"""
		Test normalization
		"""
	
		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		lnPI_1 = copy.copy(hist.data['ln(PI)'])
		self.assertTrue(np.abs(np.sum(np.exp(hist.data['ln(PI)'])) - 1.0) > 1.0e-6)
		fail = False
		try:
			hist.normalize()
		except Exception as e:
			print e
			fail = True	
		self.assertTrue(not fail)
		self.assertTrue(np.abs(np.sum(np.exp(hist.data['ln(PI)'])) - 1.0) < 1.0e-6)
	
	def testRew(self):
		"""
		Test reweighting
		"""

		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		lnPI_1 = copy.copy(hist.data['ln(PI)'])
		
		# initial reweight
		fail = False
		try:
			hist.reweight(0.0)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		lnPI_2 = copy.copy(hist.data['ln(PI)'])
		
		x = lnPI_1+np.arange(0,31)*self.beta_ref*(0.0-self.mu_ref[0])
		x -= np.log(np.sum(np.exp(x)))
		self.assertTrue(np.all(np.abs(lnPI_2-x) < 1.0e-12))
		
		# reweight again with the modified data
		fail = False
		try:
			hist.reweight(-5.0)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		lnPI_3 = copy.copy(hist.data['ln(PI)'])
		
		x = lnPI_1+np.arange(0,31)*self.beta_ref*(-5.0-self.mu_ref[0])
		x -= np.log(np.sum(np.exp(x)))
		self.assertTrue(np.all(np.abs(lnPI_3-x) < 1.0e-12))

		# start over and reweight to check consistenccy
		hist.clear()
		hist.reload()
		self.assertTrue(np.all(np.abs(hist.data['ln(PI)'] - lnPI_1)) < 1.0e-12)
		fail = False
		try:
			hist.reweight(-5.0)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(np.all(np.abs(hist.data['ln(PI)']-lnPI_3) < 1.0e-12))
	
	def testRelextrema(self):
		"""
		Test surface identification of local max/min
		"""

		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		hist.data['ln(PI)'] = np.array([1,2,3,2,1,2,3,4,5])
		
		fail = False
		try:
			hist.relextrema()
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(np.all(hist.data['ln(PI)_maxima_idx'] == [2,8]))
		self.assertTrue(np.all(hist.data['ln(PI)_minima_idx'] == [0,4]))
		
		hist.data['ln(PI)'] = np.array([1,2,3,2,1,2])
		fail = False
		try:
			hist.relextrema()
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(np.all(hist.data['ln(PI)_maxima_idx'] == [2,5]))
		self.assertTrue(np.all(hist.data['ln(PI)_minima_idx'] == [0,4]))
		
		hist.data['ln(PI)'] = np.array([1,2,3,2,1])
		fail = False
		try:
			hist.relextrema()
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(np.all(hist.data['ln(PI)_maxima_idx'] == [2]))
		self.assertTrue(np.all(hist.data['ln(PI)_minima_idx'] == [0,4]))
		
		hist.data['ln(PI)'] = np.array([2,1,2,3,2,1])
		fail = False
		try:
			hist.relextrema()
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(np.all(hist.data['ln(PI)_maxima_idx'] == [0,3]))
		self.assertTrue(np.all(hist.data['ln(PI)_minima_idx'] == [1,5]))
		
	def testThermo(self):
		"""
		Test thermo calculations for phases
		"""

		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		hist.data['mom'] = np.ones((2,3,2,3,3,31), dtype=np.float64)
		hist.data['ln(PI)'] = np.array([0,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0])
		hist.data['mom'][0,1,0,0,:] = np.arange(0,31)
		hist.data['mom'][1,1,0,0,:] = np.arange(0,31)*2
		
		fail = False
		try:
			hist.thermo()
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		
		self.assertTrue(len(hist.data['thermo']) == 2)
		self.assertTrue(np.all(hist.data['ln(PI)_maxima_idx'] == [10,25]))
		self.assertTrue(np.abs(hist.data['thermo'][0]['F.E./kT'] - -np.log(np.sum(np.exp(hist.data['ln(PI)'][:20] - hist.data['ln(PI)'][0])))) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][1]['F.E./kT'] - -np.log(np.sum(np.exp(hist.data['ln(PI)'][20:] - hist.data['ln(PI)'][0])))) < 1.0e-6)
		
		self.assertTrue(np.abs(np.sum(np.exp(hist.data['ln(PI)'][:20])*np.arange(0,20))/np.sum(np.exp(hist.data['ln(PI)'][:20])) - hist.data['thermo'][0]['n1']) < 1.0e-6)
		self.assertTrue(np.abs(np.sum(np.exp(hist.data['ln(PI)'][:20])*np.arange(0,20)*2)/np.sum(np.exp(hist.data['ln(PI)'][:20])) - hist.data['thermo'][0]['n2']) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][0]['n1'] - 9.99979018961) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][0]['n2'] - 19.9995803792) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][0]['ntot'] - 29.9993705688) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][0]['x1'] - 9.99979018961/29.9993705688) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][0]['x2'] - 19.9995803792/29.9993705688) < 1.0e-6)
		self.assertTrue(np.abs(np.sum(np.exp(hist.data['ln(PI)'][20:])*np.arange(20,31))/np.sum(np.exp(hist.data['ln(PI)'][20:])) - hist.data['thermo'][1]['n1']) < 1.0e-6)
		self.assertTrue(np.abs(np.sum(np.exp(hist.data['ln(PI)'][20:])*np.arange(20,31)*2)/np.sum(np.exp(hist.data['ln(PI)'][20:])) - hist.data['thermo'][1]['n2']) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][1]['n1'] - 25.0) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][1]['n2'] - 50.0) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][1]['ntot'] - 75.0) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][1]['x1'] - 25./75.) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][1]['x2'] - 50./75.) < 1.0e-6)
		
	def testThermoComplete(self):
		"""
		Test thermo calculations for complete lnPI surface
		"""

		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		hist.data['mom'] = np.ones((2,3,2,3,3,31), dtype=np.float64)
		hist.data['ln(PI)'] = np.array([0,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0])
		hist.data['mom'][0,1,0,0,:] = np.arange(0,31)
		hist.data['mom'][1,1,0,0,:] = np.arange(0,31)*2
		
		fail = False
		try:
			hist.thermo(True, True)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		
		self.assertTrue(len(hist.data['thermo']) == 1)
		self.assertTrue(np.abs(hist.data['thermo'][0]['F.E./kT'] - -np.log(np.sum(np.exp(hist.data['ln(PI)'][:] - hist.data['ln(PI)'][0])))) < 1.0e-6)
		
		self.assertTrue(np.abs(np.sum(np.exp(hist.data['ln(PI)'][:])*np.arange(0,31))/np.sum(np.exp(hist.data['ln(PI)'][:])) - hist.data['thermo'][0]['n1']) < 1.0e-6)
		self.assertTrue(np.abs(np.sum(np.exp(hist.data['ln(PI)'][:])*np.arange(0,31)*2)/np.sum(np.exp(hist.data['ln(PI)'][:])) - hist.data['thermo'][0]['n2']) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][0]['n1'] - 10.0998274444) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][0]['n2'] - 20.1996548887) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][0]['ntot'] - 30.2994823331) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][0]['x1'] - 10.0998274444/30.2994823331) < 1.0e-6)
		self.assertTrue(np.abs(hist.data['thermo'][0]['x2'] - 20.1996548887/30.2994823331) < 1.0e-6)
	
	def testIsSafe(self):
		"""
		Test check that thermo calculations are safe
		"""	
		
		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		hist.data['mom'] = np.ones((2,3,2,3,3,31), dtype=np.float64)
		hist.data['ln(PI)'] = np.array([0,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0])
		hist.data['mom'][0,1,0,0,:] = np.arange(0,31)
		hist.data['mom'][1,1,0,0,:] = np.arange(0,31)*2
		
		fail = False
		try:
			hist.thermo()
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		
		self.assertTrue(not hist.is_safe(10.0))
		self.assertTrue(hist.is_safe(5.0))
		self.assertTrue(hist.is_safe(10.0, True))
		self.assertTrue(not hist.is_safe(10.1, True))
				 
	def testPhaseEq(self):
		"""
		Test phase equilibrium at this temperature
		"""

		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		hist.data['ln(PI)'] = np.array([0,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0])
	
		fail = False
		try:
			eq_hist = hist.find_phase_eq(0.001, self.mu_ref[0])
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(np.abs(eq_hist.data['thermo'][0]['F.E./kT']-eq_hist.data['thermo'][1]['F.E./kT']) < 0.001)
	
	def testTempExtrap1(self):
		"""
		Test first order temperature extrapolation
		"""	
	
		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		hist.data['mom'] = np.ones((2,3,2,3,3,31), dtype=np.float64)
		hist.data['ln(PI)'] = np.array([0,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0])
		hist.data['mom'][0,1,0,0,:] = np.arange(0,31)
		hist.data['mom'][0,1,1,0,:] = np.arange(0,31)
		hist.data['mom'][0,0,0,1,:] = np.arange(0,31)
		hist.data['mom'][1,0,0,1,:] = np.arange(0,31)
		hist.data['mom'][1,1,0,0,:] = np.arange(0,31)*2
		hist.data['mom'][1,1,1,0,:] = np.arange(0,31)*2
		hist.data['mom'][0,0,1,1,:] = np.arange(0,31)*2
		hist.data['mom'][1,0,1,1,:] = np.arange(0,31)*2
		
		hist.data['mom'][:,1,:,1,:] = 1.234*np.ones(31, dtype=np.float64)
		
		beta = 2.0*hist.data['curr_beta']
		
		hist.normalize()
		lnPI_orig = copy.copy(hist.data['ln(PI)'])
		mom = copy.copy(hist.data['mom'])
		
		f_n1n2 = hist.data['mom'][0,1,1,1,0] - hist.data['mom'][0,1,1,0,0]*hist.data['mom'][0,0,1,1,0]
		f_n2n2 = hist.data['mom'][1,1,1,1,0] - hist.data['mom'][1,1,1,0,0]*hist.data['mom'][1,0,1,1,0]
		
		mom[0,1,0,0,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n1n2
		mom[1,1,0,0,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n2n2
		mom[0,0,0,1,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n1n2
		mom[0,0,1,1,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n2n2
		
		mom[0,1,1,0,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n1n2
		mom[1,1,1,0,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n2n2
		mom[1,0,0,1,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n1n2
		mom[1,0,1,1,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n2n2
		
		ave_n1 = 10.0998274444
		ave_n2 = 20.1996548887
		ave_ntot = 30.2994823331
		ave_u = 1.0
		
		dlnPI = hist.data['curr_mu'][0]*(np.arange(0,31) - ave_ntot) + (hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*(np.arange(0,31)*2 - ave_n2) - (np.ones(31, dtype=np.float64) - ave_u)
		ans = lnPI_orig + dlnPI*(beta - hist.data['curr_beta'])
		ans -= np.log(np.sum(np.exp(ans)))
		new_hist = hist.temp_extrap(beta, 1, 10.0, True, True, True)
		
		self.assertTrue(np.all(np.abs(ans - new_hist.data['ln(PI)']) < 1.0e-12))
		self.assertTrue(np.all(np.abs(beta - new_hist.data['curr_beta']) < 1.0e-12))
	
	def testTempExtrap2(self):
		"""
		Test second order temperature extrapolation
		"""
	
		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		beta = 2.0*self.beta_ref
		
		# expect fail from low order
		fail = False
		try:
			new_hist = hist.temp_extrap(beta, 2, 10.0, True, True)
		except Exception as e:
			#print e
			fail = True
		self.assertTrue(fail)

	def testDMu2Extrap1(self):
		"""
		Test first order dMu extrapolation with 2 components
		"""

		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		target_dmu = np.array([-4.0]) # mu = [5.0, 0.0] so dMu2 = -5.0 originally

		fail = False
		try:
			 newh = hist.dmu_extrap(target_dmu, 1, 10.0, True, True, False)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)

		self.assertTrue(np.all(newh.data['curr_mu'] == [5.0, 1.0]))
		self.assertTrue(newh.data['curr_beta'] == self.beta_ref)
	
		ave_n2 = np.sum(np.exp(hist.data['ln(PI)'])*hist.data['mom'][1,1,0,0,0])/np.sum(np.exp(hist.data['ln(PI)']))

		check = hist.data['ln(PI)'] + (hist.data['curr_beta']*(hist.data['mom'][1,1,0,0,0]-ave_n2)*1.0)

		newh.normalize()
		check = np.log(np.exp(check)/np.sum(np.exp(check)))
		
		self.assertTrue(np.all(newh.data['ln(PI)']-check) < 1.0e-12)

	def testDMu2Extrap2(self):
		"""
		Test second order dMu extrapolation with 2 components
		"""

		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		target_dmu = np.array([-4.0]) # mu = [5.0, 0.0] so dMu2 = -5.0 originally

		fail = False
		try:
			 newh = hist.dmu_extrap(target_dmu, 2, 10.0, True, True, True)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(np.all(newh.data['curr_mu'] == [5.0, 1.0]))
		self.assertTrue(newh.data['curr_beta'] == self.beta_ref)
		newh.normalize()

		ave_n2 = np.sum(np.exp(hist.data['ln(PI)'])*hist.data['mom'][1,1,0,0,0])/np.sum(np.exp(hist.data['ln(PI)']))
		f_tilde = self.beta_ref*self.beta_ref*(hist.data['mom'][1,2,0,0,0] - hist.data['mom'][1,1,0,0,0]*hist.data['mom'][1,1,0,0,0])
		f_hat = self.beta_ref*self.beta_ref*(np.sum(np.exp(hist.data['ln(PI)'])*hist.data['mom'][1,2,0,0,0])/np.sum(np.exp(hist.data['ln(PI)'])) - np.sum(np.exp(hist.data['ln(PI)'])*hist.data['mom'][1,1,0,0,0])/np.sum(np.exp(hist.data['ln(PI)']))*np.sum(np.exp(hist.data['ln(PI)'])*hist.data['mom'][1,1,0,0,0])/np.sum(np.exp(hist.data['ln(PI)'])) )

		# first order
		check = hist.data['ln(PI)'] + (hist.data['curr_beta']*(hist.data['mom'][1,1,0,0,0]-ave_n2)*1.0)

		# second order
		check += 0.5*1.0*1.0*(f_tilde - f_hat)

		# normalize new estimate
		check = np.log(np.exp(check)/np.sum(np.exp(check)))
		
		self.assertTrue(np.all(newh.data['ln(PI)']-check) < 1.0e-12)

	def testTempDMu2Extrap1(self):
		"""
		Test first order T+dMu extrapolation with 2 components
		"""

		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		target_dmu = np.array([-4.0]) # mu = [5.0, 0.0] so dMu2 = -5.0 originally
		target_beta = 2.0*hist.data['curr_beta']

		fail = False
		try:
			 newh = hist.temp_dmu_extrap(target_beta, target_dmu, 1, 10.0, True, True, True)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(np.all(newh.data['curr_mu'] == [5.0, 1.0]))
		self.assertTrue(newh.data['curr_beta'] == target_beta)
		newh.normalize()

		ave_n2 = np.sum(np.exp(hist.data['ln(PI)'])*hist.data['mom'][1,1,0,0,0])/np.sum(np.exp(hist.data['ln(PI)']))
		ave_ntot = np.sum(np.exp(hist.data['ln(PI)'])*hist.data['ntot'])/np.sum(np.exp(hist.data['ln(PI)']))
		ave_u = np.sum(np.exp(hist.data['ln(PI)'])*hist.data['mom'][0,0,0,0,1])/np.sum(np.exp(hist.data['ln(PI)']))

		check = hist.data['ln(PI)'] + (hist.data['curr_beta']*(hist.data['mom'][1,1,0,0,0]-ave_n2)*1.0)
		dlnPI = hist.data['curr_mu'][0]*(hist.data['ntot'] - ave_ntot) + (hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*(hist.data['mom'][1,1,0,0,0] - ave_n2) - (hist.data['mom'][0,0,0,0,1] - ave_u)
		check += dlnPI*(target_beta - hist.data['curr_beta'])
		check -= np.log(np.sum(np.exp(check)))

		self.assertTrue(np.all(newh.data['ln(PI)']-check) < 1.0e-12)		
	
	def testTempDMu2Extrap2(self):
		"""
		Test second order T+dMu extrapolation with 2 components
		"""
	
		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth)
		target_dmu = np.array([-4.0]) # mu = [5.0, 0.0] so dMu2 = -5.0 originally
		target_beta = 2.0*hist.data['curr_beta']

		fail = False
		try:
			 newh = hist.temp_dmu_extrap(target_beta, target_dmu, 2, 10.0, True, True, True)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(np.all(newh.data['curr_mu'] == [5.0, 1.0]))
		self.assertTrue(newh.data['curr_beta'] == target_beta)
		newh.normalize()

		# first order corrections
		ave_n2 = np.sum(np.exp(hist.data['ln(PI)'])*hist.data['mom'][1,1,0,0,0])/np.sum(np.exp(hist.data['ln(PI)']))
		ave_ntot = np.sum(np.exp(hist.data['ln(PI)'])*hist.data['ntot'])/np.sum(np.exp(hist.data['ln(PI)']))
		ave_u = np.sum(np.exp(hist.data['ln(PI)'])*hist.data['mom'][0,0,0,0,1])/np.sum(np.exp(hist.data['ln(PI)']))

		check = hist.data['ln(PI)'] + (hist.data['curr_beta']*(hist.data['mom'][1,1,0,0,0]-ave_n2)*1.0)
		dlnPI = hist.data['curr_mu'][0]*(hist.data['ntot'] - ave_ntot) + (hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*(hist.data['mom'][1,1,0,0,0] - ave_n2) - (hist.data['mom'][0,0,0,0,1] - ave_u)
		check += dlnPI*(target_beta - hist.data['curr_beta'])

		# second order corrections
		H = np.zeros((2,2,len(hist.data['ntot'])), dtype=np.float64)
		h = np.zeros((2,2), dtype=np.float64)
		xi = np.array([(target_beta - hist.data['curr_beta']),1.0])

		H[0,0] = -hist.data['curr_mu'][0]*hist._gc_dX_dB([0,0,0,0,0],1) + (hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*(hist._sg_dX_dB([1,1,0,0,0],0) - hist._gc_dX_dB([1,1,0,0,0],0)) - (hist._sg_dX_dB([0,0,0,0,1],0) - hist._gc_dX_dB([0,0,0,0,1],0))
		H[0,1] = (hist.data['mom'][1,1,0,0,0] - ave_n2) + hist.data['curr_beta']*(hist._sg_dX_dB([1,1,0,0,0],0) - hist._gc_dX_dB([1,1,0,0,0],0))
		H[1,0] = copy.copy(H[0][1])
		f_t = hist.data['mom'][1,2,0,0,0] - hist.data['mom'][1,1,0,0,0]**2
		f_h = hist._gc_fluct_ii([1,1,0,0,0],[1,1,0,0,0])
		H[1,1] = hist.data['curr_beta']**2*(f_t - f_h)

		# do this loop VERY manually to be sure the 'tricks' in actual pyx file are correct
		for i in xrange(0, len(hist.data['ntot'])):
			h[0,0] = H[0,0,i]
			h[1,0] = H[1,0,i]
			h[0,1] = H[0,1,i]
			h[1,1] = H[1,1,i]
			check[i] += 0.5*np.sum(np.dot(xi,h)*xi)

		check -= np.max(check)
		check -= np.log(np.sum(np.exp(check)))
		self.assertTrue(np.all(newh.data['ln(PI)']-check) < 1.0e-12)
	
	def testTempExtrap1KE(self):
		"""
		Test first order temperature extrapolation when KE contributions are present. This should be identical to the case when we say no KE present since dlnPI_dB is structurally the same
		"""

		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, True)
		self.assertTrue(hist.metadata['used_ke'])

		hist.data['mom'] = np.ones((2,3,2,3,3,31), dtype=np.float64)
		hist.data['ln(PI)'] = np.array([0,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0])
		hist.data['mom'][0,1,0,0,:] = np.arange(0,31)
		hist.data['mom'][0,1,1,0,:] = np.arange(0,31)
		hist.data['mom'][0,0,0,1,:] = np.arange(0,31)
		hist.data['mom'][1,0,0,1,:] = np.arange(0,31)
		hist.data['mom'][1,1,0,0,:] = np.arange(0,31)*2
		hist.data['mom'][1,1,1,0,:] = np.arange(0,31)*2
		hist.data['mom'][0,0,1,1,:] = np.arange(0,31)*2
		hist.data['mom'][1,0,1,1,:] = np.arange(0,31)*2
		
		hist.data['mom'][:,1,:,1,:] = 1.234*np.ones(31, dtype=np.float64)
		
		beta = 2.0*hist.data['curr_beta']
		
		hist.normalize()
		lnPI_orig = copy.copy(hist.data['ln(PI)'])
		mom = copy.copy(hist.data['mom'])
		
		f_n1n2 = hist.data['mom'][0,1,1,1,0] - hist.data['mom'][0,1,1,0,0]*hist.data['mom'][0,0,1,1,0]
		f_n2n2 = hist.data['mom'][1,1,1,1,0] - hist.data['mom'][1,1,1,0,0]*hist.data['mom'][1,0,1,1,0]
		
		mom[0,1,0,0,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n1n2
		mom[1,1,0,0,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n2n2
		mom[0,0,0,1,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n1n2
		mom[0,0,1,1,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n2n2
		
		mom[0,1,1,0,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n1n2
		mom[1,1,1,0,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n2n2
		mom[1,0,0,1,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n1n2
		mom[1,0,1,1,0] += (beta-self.beta_ref)*(hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*f_n2n2
		
		ave_n1 = 10.0998274444
		ave_n2 = 20.1996548887
		ave_ntot = 30.2994823331
		ave_u = 1.0
		
		dlnPI = hist.data['curr_mu'][0]*(np.arange(0,31) - ave_ntot) + (hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*(np.arange(0,31)*2 - ave_n2) - (np.ones(31, dtype=np.float64) - ave_u)
		ans = lnPI_orig + dlnPI*(beta - hist.data['curr_beta'])
		ans -= np.log(np.sum(np.exp(ans)))
		new_hist = hist.temp_extrap(beta, 1, 10.0, True, True, True)
		
		self.assertTrue(np.all(np.abs(ans - new_hist.data['ln(PI)']) < 1.0e-12))
		self.assertTrue(np.all(np.abs(beta - new_hist.data['curr_beta']) < 1.0e-12))

	def testTempExtrap2KE(self):
		"""
		Test second order temperature extrapolation when KE contributions are present.  This should be identical to the case when we say no KE present since dlnPI_dB is structurally the same
		"""

		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, True)
		beta = 2.0*self.beta_ref
		
		self.assertTrue(hist.metadata['used_ke'])

		# expect fail from low order
		fail = False
		try:
			new_hist = hist.temp_extrap(beta, 2, 10.0, True, True)
		except Exception as e:
			#print e
			fail = True
		self.assertTrue(fail)

	def testTempDMu2Extrap2KE(self):
		"""
		Test second order T+dMu extrapolation with 2 components when KE contributions are present. This should be identical to the case when we say no KE present since dlnPI_dB is structurally the same
		"""

		hist = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, True)
		target_dmu = np.array([-4.0]) # mu = [5.0, 0.0] so dMu2 = -5.0 originally
		target_beta = 2.0*hist.data['curr_beta']

		self.assertTrue(hist.metadata['used_ke'])

		fail = False
		try:
			 newh = hist.temp_dmu_extrap(target_beta, target_dmu, 2, 10.0, True, True, True)
		except Exception as e:
			print e
			fail = True
		self.assertTrue(not fail)
		self.assertTrue(np.all(newh.data['curr_mu'] == [5.0, 1.0]))
		self.assertTrue(newh.data['curr_beta'] == target_beta)
		newh.normalize()

		# first order corrections
		ave_n2 = np.sum(np.exp(hist.data['ln(PI)'])*hist.data['mom'][1,1,0,0,0])/np.sum(np.exp(hist.data['ln(PI)']))
		ave_ntot = np.sum(np.exp(hist.data['ln(PI)'])*hist.data['ntot'])/np.sum(np.exp(hist.data['ln(PI)']))
		ave_u = np.sum(np.exp(hist.data['ln(PI)'])*hist.data['mom'][0,0,0,0,1])/np.sum(np.exp(hist.data['ln(PI)']))

		check = hist.data['ln(PI)'] + (hist.data['curr_beta']*(hist.data['mom'][1,1,0,0,0]-ave_n2)*1.0)
		dlnPI = hist.data['curr_mu'][0]*(hist.data['ntot'] - ave_ntot) + (hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*(hist.data['mom'][1,1,0,0,0] - ave_n2) - (hist.data['mom'][0,0,0,0,1] - ave_u)
		check += dlnPI*(target_beta - hist.data['curr_beta'])

		# second order corrections
		H = np.zeros((2,2,len(hist.data['ntot'])), dtype=np.float64)
		h = np.zeros((2,2), dtype=np.float64)
		xi = np.array([(target_beta - hist.data['curr_beta']),1.0])

		H[0,0] = -hist.data['curr_mu'][0]*hist._gc_dX_dB([0,0,0,0,0],1) + (hist.data['curr_mu'][1] - hist.data['curr_mu'][0])*(hist._sg_dX_dB([1,1,0,0,0],0) - hist._gc_dX_dB([1,1,0,0,0],0)) - (hist._sg_dX_dB([0,0,0,0,1],0) - hist._gc_dX_dB([0,0,0,0,1],0))
		H[0,1] = (hist.data['mom'][1,1,0,0,0] - ave_n2) + hist.data['curr_beta']*(hist._sg_dX_dB([1,1,0,0,0],0) - hist._gc_dX_dB([1,1,0,0,0],0))
		H[1,0] = copy.copy(H[0][1])
		f_t = hist.data['mom'][1,2,0,0,0] - hist.data['mom'][1,1,0,0,0]**2
		f_h = hist._gc_fluct_ii([1,1,0,0,0],[1,1,0,0,0])
		H[1,1] = hist.data['curr_beta']**2*(f_t - f_h)

		# do this loop VERY manually to be sure the 'tricks' in actual pyx file are correct
		for i in xrange(0, len(hist.data['ntot'])):
			h[0,0] = H[0,0,i]
			h[1,0] = H[1,0,i]
			h[0,1] = H[0,1,i]
			h[1,1] = H[1,1,i]
			check[i] += 0.5*np.sum(np.dot(xi,h)*xi)

		check -= np.max(check)
		check -= np.log(np.sum(np.exp(check)))
		self.assertTrue(np.all(newh.data['ln(PI)']-check) < 1.0e-12)

	def testDlnPI1KE(self):
		"""
		Test the differences between first order derivatives of lnPI with KE is/not used
		"""

		hist_ke = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, True)
		hist_pe = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, False)

		dlnPI_dB_ke, dm_dB_ke = hist_ke._dB()
		dlnPI_dB_pe, dm_dB_pe = hist_pe._dB()

		self.assertTrue(np.all(np.abs(dlnPI_dB_ke-dlnPI_dB_pe)) < 1.0e-12)

	def testDlnPI2KE(self):
		"""
		Test the differences between first order derivatives of lnPI with KE is/not used
		"""

		hist_ke = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, True)
		hist_pe = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, False)

		dlnPI_dB_ke, dm_dB_ke = hist_ke._dB2()
		dlnPI_dB_pe, dm_dB_pe = hist_pe._dB2()

		ave_ntot = np.sum(np.exp(hist_pe.data['ln(PI)'])*hist_pe.data['ntot'])/np.sum(np.exp(hist_pe.data['ln(PI)']))
		self.assertTrue(np.all(np.abs((dlnPI_dB_ke-dlnPI_dB_pe) - (1.5/self.beta_ref/self.beta_ref*(hist_pe.data['ntot']-ave_ntot)))) < 1.0e-12)

	def testSgDXKE(self):
		"""
		Test the differences between first order derivatives of extensive properties with KE is/not used (semigrand ensemble)
		"""

		hist_ke = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, True)
		hist_pe = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, False)

		# energy should be affected by KE - try different X's containing U
		d_ke = hist_ke._sg_dX_dB([0,0,0,0,1],0)
		d_pe = hist_pe._sg_dX_dB([0,0,0,0,1],0)
		x = 1.5*1/self.beta_ref/self.beta_ref*hist_pe.data['ntot']
		self.assertTrue(np.all(np.abs((d_pe-d_ke) - x)) < 1.0e-12)

		d_ke = hist_ke._sg_dX_dB([0,1,0,0,1],0)
		d_pe = hist_pe._sg_dX_dB([0,1,0,0,1],0)
		x = 1.5*1/self.beta_ref/self.beta_ref*hist_pe.data['ntot']*hist_pe.data['mom'][0,1,0,0,0]
		self.assertTrue(np.all(np.abs((d_pe-d_ke) - x)) < 1.0e-12)

		d_ke = hist_ke._sg_dX_dB([0,1,0,1,1],0)
		d_pe = hist_pe._sg_dX_dB([0,1,0,1,1],0)
		x = 1.5*1/self.beta_ref/self.beta_ref*hist_pe.data['ntot']*hist_pe.data['mom'][0,1,0,1,0]
		self.assertTrue(np.all(np.abs((d_pe-d_ke) - x)) < 1.0e-12)

		d_ke = hist_ke._sg_dX_dB([1,1,0,1,1],0)
		d_pe = hist_pe._sg_dX_dB([1,1,0,1,1],0)
		x = 1.5*1/self.beta_ref/self.beta_ref*hist_pe.data['ntot']*hist_pe.data['mom'][1,1,0,1,0]
		self.assertTrue(np.all(np.abs((d_pe-d_ke) - x)) < 1.0e-12)

		d_ke = hist_ke._sg_dX_dB([1,1,0,1,1],1)
		d_pe = hist_pe._sg_dX_dB([1,1,0,1,1],1)
		x = 1.5*1/self.beta_ref/self.beta_ref*hist_pe.data['ntot']*hist_pe.data['ntot']*hist_pe.data['mom'][1,1,0,1,0]
		self.assertTrue(np.all(np.abs((d_pe-d_ke) - x)) < 1.0e-12)

		# pk number unaffected by KE
		d_ke = hist_ke._sg_dX_dB([0,1,0,0,0],0)
		d_pe = hist_pe._sg_dX_dB([0,1,0,0,0],0)
		self.assertTrue(np.all(np.abs((d_pe-d_ke))) < 1.0e-12)

		d_ke = hist_ke._sg_dX_dB([0,1,0,1,0],0)
		d_pe = hist_pe._sg_dX_dB([0,1,0,1,0],0)
		self.assertTrue(np.all(np.abs((d_pe-d_ke))) < 1.0e-12)

		d_ke = hist_ke._sg_dX_dB([0,1,1,1,0],0)
		d_pe = hist_pe._sg_dX_dB([0,1,1,1,0],0)
		self.assertTrue(np.all(np.abs((d_pe-d_ke))) < 1.0e-12)

	def testGcDXKE(self):
		"""
		Test the differences between first order derivatives of extensive properties with KE is/not used (grand canonical ensemble)
		"""

		hist_ke = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, True)
		hist_pe = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, False)

		# energy should be affected by KE - try different X's containing U
		d_ke = hist_ke._gc_dX_dB([0,0,0,0,1],0)
		d_pe = hist_pe._gc_dX_dB([0,0,0,0,1],0)
		ave = np.sum(np.exp(hist_pe.data['ln(PI)'])*hist_pe.data['mom'][0,0,0,0,1-1]*hist_pe.data['ntot'])/np.sum(np.exp(hist_pe.data['ln(PI)']))
		x = 1.5*1/self.beta_ref/self.beta_ref*ave
		self.assertTrue(np.abs((d_pe-d_ke) - x) < 1.0e-12)

		d_ke = hist_ke._gc_dX_dB([0,1,0,0,1],0)
		d_pe = hist_pe._gc_dX_dB([0,1,0,0,1],0)
		ave = np.sum(np.exp(hist_pe.data['ln(PI)'])*hist_pe.data['mom'][0,1,0,0,1-1]*hist_pe.data['ntot'])/np.sum(np.exp(hist_pe.data['ln(PI)']))
		x = 1.5*1/self.beta_ref/self.beta_ref*ave
		self.assertTrue(np.abs((d_pe-d_ke) - x) < 1.0e-12)

		d_ke = hist_ke._gc_dX_dB([0,1,0,1,1],0)
		d_pe = hist_pe._gc_dX_dB([0,1,0,1,1],0)
		ave = np.sum(np.exp(hist_pe.data['ln(PI)'])*hist_pe.data['mom'][0,1,0,1,1-1]*hist_pe.data['ntot'])/np.sum(np.exp(hist_pe.data['ln(PI)']))
		x = 1.5*1/self.beta_ref/self.beta_ref*ave
		self.assertTrue(np.abs((d_pe-d_ke) - x) < 1.0e-12)

		d_ke = hist_ke._gc_dX_dB([1,1,0,1,1],0)
		d_pe = hist_pe._gc_dX_dB([1,1,0,1,1],0)
		ave = np.sum(np.exp(hist_pe.data['ln(PI)'])*hist_pe.data['mom'][1,1,0,1,1-1]*hist_pe.data['ntot'])/np.sum(np.exp(hist_pe.data['ln(PI)']))
		x = 1.5*1/self.beta_ref/self.beta_ref*ave
		self.assertTrue(np.abs((d_pe-d_ke) - x) < 1.0e-12)

		d_ke = hist_ke._gc_dX_dB([1,1,0,1,1],1)
		d_pe = hist_pe._gc_dX_dB([1,1,0,1,1],1)
		ave = np.sum(np.exp(hist_pe.data['ln(PI)'])*hist_pe.data['mom'][1,1,0,1,1-1]*hist_pe.data['ntot']*hist_pe.data['ntot'])/np.sum(np.exp(hist_pe.data['ln(PI)']))
		x = 1.5*1/self.beta_ref/self.beta_ref*ave
		self.assertTrue(np.abs((d_pe-d_ke) - x) < 1.0e-12)

		# pk number unaffected by KE
		d_ke = hist_ke._gc_dX_dB([0,1,0,0,0],0)
		d_pe = hist_pe._gc_dX_dB([0,1,0,0,0],0)
		self.assertTrue(np.abs((d_pe-d_ke)) < 1.0e-12)

		d_ke = hist_ke._gc_dX_dB([0,1,0,1,0],0)
		d_pe = hist_pe._gc_dX_dB([0,1,0,1,0],0)
		self.assertTrue(np.abs((d_pe-d_ke)) < 1.0e-12)

		d_ke = hist_ke._gc_dX_dB([0,1,1,1,0],0)
		d_pe = hist_pe._gc_dX_dB([0,1,1,1,0],0)
		self.assertTrue(np.abs((d_pe-d_ke)) < 1.0e-12)

		d_ke = hist_ke._gc_dX_dB([0,1,1,1,0],1)
		d_pe = hist_pe._gc_dX_dB([0,1,1,1,0],1)
		self.assertTrue(np.abs((d_pe-d_ke)) < 1.0e-12)

	def testSgD2XKE(self):
		"""
		Test the differences between second order derivatives of extensive properties with KE is/not used (semigrand ensemble)
		"""

		fname = 'reference/test2.nc'
		hist_ke = oneDH.histogram(fname, self.beta_ref, self.mu_ref, self.smooth, True)
		hist_pe = oneDH.histogram(fname, self.beta_ref, self.mu_ref, self.smooth, False)

		# energy should be affected by KE - try different X's containing U
		d_ke = hist_ke._sg_d2X_dB2([0,0,0,0,1],0)
		d_pe = hist_pe._sg_d2X_dB2([0,0,0,0,1],0)
		x = 1.5*1/self.beta_ref/self.beta_ref*hist_pe.data['ntot']*(-2.0/self.beta_ref*hist_pe.data['mom'][0,0,0,0,0]+hist_pe._sg_dX_dB([0,0,0,0,0],0))
		self.assertTrue(np.all(np.abs((d_pe-d_ke) - x)) < 1.0e-12)

		d_ke = hist_ke._sg_d2X_dB2([0,1,0,0,1],0)
		d_pe = hist_pe._sg_d2X_dB2([0,1,0,0,1],0)
		x = 1.5*1/self.beta_ref/self.beta_ref*hist_pe.data['ntot']*(-2.0/self.beta_ref*hist_pe.data['mom'][0,1,0,0,0]+hist_pe._sg_dX_dB([0,1,0,0,0],0))
		self.assertTrue(np.all(np.abs((d_pe-d_ke) - x)) < 1.0e-12)

		d_ke = hist_ke._sg_d2X_dB2([0,1,0,1,1],0)
		d_pe = hist_pe._sg_d2X_dB2([0,1,0,1,1],0)
		x = 1.5*1/self.beta_ref/self.beta_ref*hist_pe.data['ntot']*(-2.0/self.beta_ref*hist_pe.data['mom'][0,1,0,1,0]+hist_pe._sg_dX_dB([0,1,0,1,0],0))
		self.assertTrue(np.all(np.abs((d_pe-d_ke) - x)) < 1.0e-12)

		d_ke = hist_ke._sg_dX_dB([1,1,0,1,1],0)
		d_pe = hist_pe._sg_dX_dB([1,1,0,1,1],0)
		x = 1.5*1/self.beta_ref/self.beta_ref*hist_pe.data['ntot']*(-2.0/self.beta_ref*hist_pe.data['mom'][1,1,0,1,0]+hist_pe._sg_dX_dB([1,1,0,1,0],0))
		self.assertTrue(np.all(np.abs((d_pe-d_ke) - x)) < 1.0e-12)

		d_ke = hist_ke._sg_dX_dB([1,1,0,1,1],1)
		d_pe = hist_pe._sg_dX_dB([1,1,0,1,1],1)
		x = 1.5*1/self.beta_ref/self.beta_ref*hist_pe.data['ntot']*(-2.0/self.beta_ref*hist_pe.data['mom'][1,1,0,1,0]*hist_pe.data['ntot']+hist_pe._sg_dX_dB([1,1,0,1,0],1))
		self.assertTrue(np.all(np.abs((d_pe-d_ke) - x)) < 1.0e-12)

		# pk number unaffected by KE
		d_ke = hist_ke._sg_d2X_dB2([0,1,0,0,0],0)
		d_pe = hist_pe._sg_d2X_dB2([0,1,0,0,0],0)
		self.assertTrue(np.all(np.abs((d_pe-d_ke))) < 1.0e-12)

		d_ke = hist_ke._sg_d2X_dB2([0,1,0,1,0],0)
		d_pe = hist_pe._sg_d2X_dB2([0,1,0,1,0],0)
		self.assertTrue(np.all(np.abs((d_pe-d_ke))) < 1.0e-12)

		d_ke = hist_ke._sg_d2X_dB2([0,1,1,1,0],0)
		d_pe = hist_pe._sg_d2X_dB2([0,1,1,1,0],0)
		self.assertTrue(np.all(np.abs((d_pe-d_ke))) < 1.0e-12)

		d_ke = hist_ke._sg_d2X_dB2([0,1,1,1,0],1)
		d_pe = hist_pe._sg_d2X_dB2([0,1,1,1,0],1)
		self.assertTrue(np.all(np.abs((d_pe-d_ke))) < 1.0e-12)

	def testGcD2XKE(self):
		"""
		Test the differences between second order derivatives of extensive properties with KE is/not used (grand canonical ensemble)
		"""

		hist_ke = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, True)
		hist_pe = oneDH.histogram(self.fname, self.beta_ref, self.mu_ref, self.smooth, False)

		# energy should be affected by KE - try different X's containing U
		idx, n = [0,0,0,0,1],0
		d_pe, d_ke, x, d1, d2, d3 = compareGcD2X(hist_ke, hist_pe, idx, n, self.beta_ref, self.mu_ref)
		self.assertTrue(np.abs((d_pe-d_ke) - (x-(d1-d2+d3))) < 1.0e-12)

		idx, n = [0,1,0,0,1],0
		d_pe, d_ke, x, d1, d2, d3 = compareGcD2X(hist_ke, hist_pe, idx, n, self.beta_ref, self.mu_ref)
		self.assertTrue(np.abs((d_pe-d_ke) - (x-(d1-d2+d3))) < 1.0e-12)

		idx, n = [0,1,0,1,1],0
		d_pe, d_ke, x, d1, d2, d3 = compareGcD2X(hist_ke, hist_pe, idx, n, self.beta_ref, self.mu_ref)
		self.assertTrue(np.abs((d_pe-d_ke) - (x-(d1-d2+d3))) < 1.0e-12)

		idx, n = [1,1,0,1,1],0
		d_pe, d_ke, x, d1, d2, d3 = compareGcD2X(hist_ke, hist_pe, idx, n, self.beta_ref, self.mu_ref)
		self.assertTrue(np.abs((d_pe-d_ke) - (x-(d1-d2+d3))) < 1.0e-12)

		idx, n = [1,1,0,1,1],1
		d_pe, d_ke, x, d1, d2, d3 = compareGcD2X(hist_ke, hist_pe, idx, n, self.beta_ref, self.mu_ref)
		self.assertTrue(np.abs((d_pe-d_ke) - (x-(d1-d2+d3))) < 1.0e-12)
		
		# pk number unaffected by KE

		idx, n = [0,1,0,0,0],0
		d_pe, d_ke, x, d1, d2, d3 = compareGcD2X(hist_ke, hist_pe, idx, n, self.beta_ref, self.mu_ref)
		self.assertTrue(np.abs((d_pe-d_ke) - (0.0-(d1-d2+d3))) < 1.0e-12)

		idx, n = [0,1,0,1,0],0
		d_pe, d_ke, x, d1, d2, d3 = compareGcD2X(hist_ke, hist_pe, idx, n, self.beta_ref, self.mu_ref)
		self.assertTrue(np.abs((d_pe-d_ke) - (0.0-(d1-d2+d3))) < 1.0e-12)

		idx, n = [0,1,1,1,0],0
		d_pe, d_ke, x, d1, d2, d3 = compareGcD2X(hist_ke, hist_pe, idx, n, self.beta_ref, self.mu_ref)
		self.assertTrue(np.abs((d_pe-d_ke) - (0.0-(d1-d2+d3))) < 1.0e-12)

		idx, n = [0,1,1,1,0],1
		d_pe, d_ke, x, d1, d2, d3 = compareGcD2X(hist_ke, hist_pe, idx, n, self.beta_ref, self.mu_ref)
		self.assertTrue(np.abs((d_pe-d_ke) - (0.0-(d1-d2+d3))) < 1.0e-12)


def compareGcD2X(hist_ke, hist_pe, idx, n, beta_ref, mu_ref):
	"""
	Compare semigrand second derivative when using kinetic contributions.

	Parameters
	----------
	hist_ke : histogram
		Histogram object where potential energy + kinetic energy stored during simulation
	hist_pe : histogram
		Histogram object where potential energy stored during simulation
	idx : array
		Matrix index of property to examine
	n : int
		Exponent on N_tot
	beta_ref : double
		1/T of simulation
	mu_ref : array
		Reference state chemical potential of each component (where simulation performed at)

	"""

	idx_s = copy.copy(idx)
	idx_s[4] -= 1
	d_ke = hist_ke._gc_d2X_dB2(idx,n)
	d_pe = hist_pe._gc_d2X_dB2(idx,n)
	ave = np.sum(np.exp(hist_ke.data['ln(PI)'])*hist_ke.data['mom'][idx[0],idx[1],idx[2],idx[3],idx[4]-1]*hist_ke.data['ntot']**(n+1))/np.sum(np.exp(hist_ke.data['ln(PI)']))
	a = -2.0/beta_ref*ave
	b = hist_ke._gc_dX_dB(idx_s,n+1)
	x = 1.5*1/beta_ref/beta_ref*(a + b)

	d1 = hist_ke._gc_df_dB_ii((idx,n), ([1,1,0,0,0],0))*(mu_ref[1]-mu_ref[0])
	d1 -= hist_pe._gc_df_dB_ii((idx,n), ([1,1,0,0,0],0))*(mu_ref[1]-mu_ref[0])
	d2 = hist_ke._gc_df_dB_ii((idx,n),([0,0,0,0,1],0))
	d2 -= hist_pe._gc_df_dB_ii((idx,n),([0,0,0,0,1],0))
	d3 = mu_ref[0]*hist_ke._gc_df_dB_in((idx,n),1)
	d3 -= mu_ref[0]*hist_pe._gc_df_dB_in((idx,n),1)

	return d_pe, d_ke, x, d1, d2, d3
	
if __name__ == '__main__':
	unittest.main()
	
	"""

	* To Do:

	Should unittest more components that 2 when this is going to be used.

	* Notes:

	Third order T extrapolation has been "tested" with real data, even though not unittested here

	"""
