#BUILD:  python setup.py build_ext --inplace

from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from Cython.Distutils import build_ext
import numpy as np

omcs_patch = Extension("omcs_patch", ["omcs_patch.pyx"],
	include_dirs=[np.get_include()],libraries=["m"],)

omcs_equil = Extension("omcs_equil", ["omcs_equil.pyx"],
	include_dirs=[np.get_include()],libraries=["m"],)

chkpt_equil = Extension("chkpt_equil", ["chkpt_equil.pyx"],
	include_dirs=[np.get_include()],libraries=["m"],)

chkpt_patch = Extension("chkpt_patch", ["chkpt_patch.pyx"],
	include_dirs=[np.get_include()],libraries=["m"],)

setup(cmdclass={'build_ext': build_ext},
	ext_modules=[omcs_patch, omcs_equil, chkpt_equil, chkpt_patch])
