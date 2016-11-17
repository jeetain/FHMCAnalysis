#BUILD:  python setup.py build_ext --inplace

from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from Cython.Distutils import build_ext
import numpy as np

gc_hist = Extension("gc_hist", ["gc_hist.pyx"], include_dirs=[np.get_include()],libraries=["m"],)
setup(cmdclass={'build_ext': build_ext}, ext_modules=[gc_hist])