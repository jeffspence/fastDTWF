import setuptools
import sys

sys.path[0:0] = ['fastDTWF']
from package_metadata import *

setuptools.setup(
    name=NAME,
    version=VERSION,
    description='',
    author=AUTHOR,
    author_email=EMAIL,
    packages=['fastDTWF'],
    package_dir={'fastDTWF': 'fastDTWF'},
    install_requires=['numpy>=1.20.0',
                      'scipy>=1.7.3',
                      'torch>=1.13.1',
                      'numba>=0.55.2'],
)
