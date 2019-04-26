from distutils.core import setup, Extension
from Cython.Build import cythonize
import os


setup(name='cython test', ext_modules=cythonize('houghline.pyx'))
os.system('mv build/lib.linux-x86_64-3.6/my_cython/*.so ./')
