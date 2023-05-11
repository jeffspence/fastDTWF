from fastDTWF import VERSION
import setuptools


setuptools.setup(
    name='fastDTWF',
    version=VERSION,
    description='',
    author='Jeffrey P. Spence',
    author_email='jspence@stanford.edu',
    packages=['fastDTWF'],
    package_dir={'fastDTWF': 'fastDTWF'},
    install_requires=['numpy>=1.20.0',
                      'scipy>=1.7.3',
                      'torch>=1.13.1',
                      'numba>=0.55.2'],
)
