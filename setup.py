from setuptools import setup, Extension
from Cython.Distutils import build_ext
import sys
import numpy

major, minor1, minor2, release, serial = sys.version_info

if (major < 2) or (major == 2 and minor1 < 7):
    raise SystemExit("mbb_emcee requires Python 2.7 or later")

ext_modules = [Extension("fnu", ["mbb_emcee/fnu.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=["m"])]

setup(
    name="mbb_emcee",
    version="0.5.2",
    provides="mbb_emcee",
    author="Alexander Conley",
    author_email="alexander.conley@colorado.edu",
    packages=["mbb_emcee", "mbb_emcee.tests"],
    package_data={'mbb_emcee': ['resources/*.txt']},
    scripts=["mbb_emcee/run_mbb_emcee.py"],
    license="GPLv2",
    description="Modified blackbody fitting using MCMC",
    classifiers=[
        "Development Status :: 3 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
    ],
    requires=['numpy (>1.7.0)', 'emcee (>1.0.0)', 'scipy (>0.8.0)',
              'astropy (>0.2.4)', 'cython (>0.11.0)', 'h5py (>2.0.0)'],
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
