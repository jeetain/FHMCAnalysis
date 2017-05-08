# Flat Histogram Monte Carlo Analysis

Nathan A. Mahynski

---

Status

[![Build Status](https://travis-ci.org/mahynski/FHMCAnalysis.svg?branch=master)](https://travis-ci.org/mahynski/FHMCAnalysis) [![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badge/) [![Code Issues](https://www.quantifiedcode.com/api/v1/project/fe2f9564d3e84213a5ecae6e84c6f1d0/badge.svg)](https://www.quantifiedcode.com/app/project/fe2f9564d3e84213a5ecae6e84c6f1d0) [![CodeFactor](https://www.codefactor.io/repository/github/mahynski/fhmcanalysis/badge)](https://www.codefactor.io/repository/github/mahynski/fhmcanalysis) [![DOI](https://zenodo.org/badge/73996052.svg)](https://zenodo.org/badge/latestdoi/73996052)

---

## Installation:

```bash
$ git clone https://github.com/mahynski/FHMCAnalysis.git
$ cd FHMCAnalysis
$ python install.py
```

---

## Unittesting

```bash
$ cd unittests
$ python run_tests.py
```

---

## Dependencies

1. h5py (http://www.h5py.org/)
2. netCDF4 (http://unidata.github.io/netcdf4-python/)
3. Cython (http://cython.org/)
4. NumPy (http://www.numpy.org/)
5. SciPy (https://www.scipy.org/)

Alternatively, I highly recommend simply installing anaconda python (https://www.continuum.io/downloads), and installing netCDF4 and Cython libraries through conda. For example:

```bash
$ cd ~/Downloads/
$ wget https://repo.continuum.io/archive/Anaconda2-4.2.0-MacOSX-x86_64.pkg
$ bash Anaconda2-4.2.0-MacOSX-x86_64.sh
```

---

## Input

+ After running simulations using [FHMCSimulation](https://mahynski.github.io/FHMCSimulation/), this will output the requisite information in each "window" directory that was performed.  In this library, moments.win_patch.fhmc_patch (or chkpt_patch) and moments.win_patch.fhmc_equil (or chkpt_equil), provide utilities for identifying consecutive windows which are sufficiently "equilibrated" for use and patching those windows together to form a single macrostate distribution, e.g. result.nc (saved as netCDF4 file).

```python
import FHMCAnalysis.moments.win_patch.fhmc_patch as wp
import FHMCAnalysis.moments.win_patch.fhmc_equil as we

seq = wp.get_patch_sequence(src='./')
seq = we.test_nebr_equil(seq, per_err=1.0) # accept max of 1% deviation between overlapping windows
wp.patch_all_windows(seq, composite='./composite.nc')
```

+ Then use histogram modules to perform reweighting, phase behavior calculations, etc. on the resulting composite.nc file.  See, e.g. moments.histogram.one_dim.ntot.gc_hist for simulations performed where N_{tot} was used as the order parameter. See example/ntot for more detailed example(s).

```python
import FHMCAnalysis.moments.histogram.one_dim.ntot.gc_hist as hg

# mu_ref = chemical potentials used during the simulations
# beta_ref = 1/T simulations performed at
# smooth = number of points in space to smooth lnPI over
hist = hg.histogram (composite='./composite.nc', beta_ref=1.0, mu_ref=[0.0], smooth=10) # create histogram

# Reweight and compute thermodynamic properties
hist.reweight(1.234) # reweight the histogram to some other mu_1
if (hist.is_safe()): # check that max(lnPI) is far enough from the edge
    hist.thermo() # compute thermodynamic properties
    print hist.data['thermo'] # results are stored here ...

# Search for phase coexistence
# lnZ_tol = error between free energies of each phase
equil_hist = hist.find_phase_eq(lnZ_tol=1.0e-5, mu_guess=1.234, beta=1.0) # search for phase equilibrium at this temperature
if (equil_hist.is_safe()): # check that max(lnPI) is far enough from the edge
    hist.thermo() # compute thermodynamic properties
    print hist.data['thermo'] # results are stored here ...
```

### Possible order parameters that can be handled

+ N_tot
+ N_1
