from distutils.core import setup

setup(
    name="modified_blackbody_emcee",
    version="1.0.0",
    author="Alexander Conley",
    author_email="alexander.conley@colorado.edu",
    packages=["modified_blackbody_emcee"],
    scripts=["modified_blackbody_emcee/run_blackbody_emcee.py"],
    license="GPL",
    description="Modified blackbody fitting using MCMC",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)

