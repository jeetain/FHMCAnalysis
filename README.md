# Flat Histogram Monte Carlo Analysis

Nathan A. Mahynski

---

Status

[![Build Status](https://travis-ci.org/mahynski/FHMCAnalysis.svg?branch=master)](https://travis-ci.org/mahynski/FHMCAnalysis) [![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badge/) [![Code Issues](https://www.quantifiedcode.com/api/v1/project/fe2f9564d3e84213a5ecae6e84c6f1d0/badge.svg)](https://www.quantifiedcode.com/app/project/fe2f9564d3e84213a5ecae6e84c6f1d0)

---

## Installation:

```
$ git clone https://github.com/mahynski/FHMCAnalysis.git
$ cd FHMCAnalysis
$ python install.py
```

---

## Dependencies

1. h5py (http://www.h5py.org/)
2. netCDF4 (http://unidata.github.io/netcdf4-python/)
3. Cython (http://cython.org/)
4. Numpy (http://www.numpy.org/)

Alternatively, I highly recommend simply installing anaconda python (https://www.continuum.io/downloads), and installing netCDF4 and Cython libraries through conda. For example:

```
$ cd ~/Downloads/
$ wget https://repo.continuum.io/archive/Anaconda2-4.2.0-MacOSX-x86_64.pkg
$ bash Anaconda2-4.2.0-MacOSX-x86_64.sh
```

---

## Unittesting

```
$ cd unittests
$ python run_tests.py
```

---

## Input

+ After running simulations using [FHMCSimulation](https://mahynski.github.io/FHMCSimulation/), this will output the requisite information in each "window" directory that was performed.  In this library, moments.win_patch.omcs_patch and moments.win_patch.omcs_equil, provide utilities for identifying consecutive windows which are sufficiently "equilibrated" for use and patching those windows together to form a single macrostate distribution, e.g. result.nc (saved as netCDF4 file).

```python
import FHMCAnalysis.moments.win_patch.omcs_patch as wp
import FHMCAnalysis.moments.win_patch.omcs_equil as we
src = './'
per_err = 1.0 # accept max of 1% deviation between overlapping windows
seq = wp.get_patch_sequence(src, per_err)
seq = we.test_nebr_equil(seq)
composite = src+'/composite.nc'
wp.patch_all_windows(seq, composite)
```

+ Then use histogram modules to perform reweighting, phase behavior calculations, etc. on the resulting composite.nc file.  See, e.g. moments.histogram.one_dim.ntot.gc_hist for simulations performed where N_{tot} was used as the order parameter. See example/ntot for a more detailed example.

```
mu_ref = [0.0] # chemical potentials used during the simulations
beta_ref = 1.0 # 1/T simulations performed at
smooth = 10 # number of points in space to smooth lnPI over
hist = histogram (composite, beta_ref, mu_ref, smooth) # create histogram

# Reweight and compute thermodynamic properties
hist.reweight(1.234) # reweight the histogram to some other mu_1
if (hist.is_safe()): # check that max(lnPI) is far enough from the edge
    hist.thermo() # compute thermodynamic properties
    print hist.data['thermo'] # results are stored here ...

# Search for phase coexistence
lnZ_tol = 1.0e-5 # error between free energies of each phase
mu_guess = 1.234
beta = beta_ref
equil_hist = hist.find_phase_eq(lnZ_tol, mu_guess, beta) # search for phase equilibrium at this temperature
if (equil_hist.is_safe()): # check that max(lnPI) is far enough from the edge
    hist.thermo() # compute thermodynamic properties
    print hist.data['thermo'] # results are stored here ...
```

### netCDF4 file format

+ N_tot
+ N_1
