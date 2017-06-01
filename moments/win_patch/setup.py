#BUILD:  python setup.py build_ext --inplace

from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from Cython.Distutils import build_ext
import numpy as np

fhmc_patch = Extension("fhmc_patch", ["fhmc_patch.pyx"],
	include_dirs=[np.get_include()], libraries=["m"],)

fhmc_equil = Extension("fhmc_equil", ["fhmc_equil.pyx"],
	include_dirs=[np.get_include()], libraries=["m"],)

chkpt_equil = Extension("chkpt_equil", ["chkpt_equil.pyx"],
	include_dirs=[np.get_include()], libraries=["m"],)

chkpt_patch = Extension("chkpt_patch", ["chkpt_patch.pyx"],
	include_dirs=[np.get_include()], libraries=["m"],)

feasst_patch = Extension("feasst_patch", ["feasst_patch.pyx"],
	include_dirs=[np.get_include()], libraries=["m"],)

feasst_equil = Extension("feasst_equil", ["feasst_equil.pyx"],
	include_dirs=[np.get_include()], libraries=["m"],)

setup(cmdclass={'build_ext': build_ext},
	ext_modules=[fhmc_patch, fhmc_equil, chkpt_equil, chkpt_patch, feasst_patch, feasst_equil])
