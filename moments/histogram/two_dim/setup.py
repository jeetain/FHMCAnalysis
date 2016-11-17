#BUILD:  python setup.py build_ext --inplace

from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from Cython.Distutils import build_ext
import numpy as np

joint_hist = Extension("joint_hist", ["joint_hist.pyx"], include_dirs=[np.get_include()],libraries=["m"],)
setup(cmdclass={'build_ext': build_ext}, ext_modules=[joint_hist])