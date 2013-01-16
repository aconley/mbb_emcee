from distutils.core import setup

import sys
major, minor1, minor2, release, serial = sys.version_info

if (major < 2) or (major == 2 and minor1 < 7):
    raise SystemExit("modified_blackbody_emcee requires Python 2.7 or later")
if (major == 3):
    raise SystemExit("modified_blackbody_emcee does not support Python 3")

setup(
    name="mbb_emcee",
    version="0.1.3",
    author="Alexander Conley",
    author_email="alexander.conley@colorado.edu",
    packages=["mbb_emcee"],
    scripts=["mbb_emcee/run_mbb_emcee.py"],
    license="GPL",
    description="Modified blackbody fitting using MCMC",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    requires = ['numpy (>1.5.0)', 'emcee (>1.0.0)', 'scipy', 
                'astropy (>0.1.0)']
)

