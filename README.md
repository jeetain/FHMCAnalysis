# Flat Histogram Monte Carlo Analysis

Nathan A. Mahynski

---

Status

[![Build Status](https://travis-ci.org/mahynski/FHMCAnalysis.svg?branch=master)](https://travis-ci.org/mahynski/FHMCAnalysis) [![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badge/)        

---

### Installation:

```
$ git clone https://github.com/mahynski/FHMCAnalysis.git
$ cd FHMCAnalysis
$ python install.py
```

---

### Dependencies

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

### Unittesting

```
$ cd unittests
$ python run_tests.py
```

---

### Expected netCDF4 file format

1. N_tot
