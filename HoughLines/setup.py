from distutils.core import setup, Extension
from Cython.Build import cythonize
import os


setup(name='cython test', ext_modules=cythonize('houghline.pyx'))
