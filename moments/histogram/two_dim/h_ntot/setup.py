#BUILD:  python setup.py build_ext --inplace

from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from Cython.Distutils import build_ext
import numpy as np

pore_hist = Extension("pore_hist", ["pore_hist.pyx"], include_dirs=[np.get_include()],libraries=["m"],)
free_energy_profile = Extension("free_energy_profile", ["free_energy_profile.pyx"], include_dirs=[np.get_include()],libraries=["m"],)
organize = Extension("organize", ["organize.pyx"], include_dirs=[np.get_include()],libraries=["m"],)

setup(cmdclass={'build_ext': build_ext}, ext_modules=[pore_hist, free_energy_profile, organize])
