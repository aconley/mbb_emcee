mbb_emcee
========================

This is a package to fit modified blackbodies to photometry
data using an affine invariant MCMC.

###Installation
The usual

	python setup.py build install

###Usage
The command line routine is

	run_mbb_emcee.py

Help can be obtained via

	run_mbb_emcee.py --help

Carrying out a fit produces a HDF5 output file containing the results,
which can either be read directly, or read back into a mbb_results
object for analysis.  For example, if the output is saved to
the file test.h5:

	import mbb_emcee
	res = mbb_emcee.mbb_results()
	res.readFromHDF5("test.h5")
	T_val = res.par_cen('T')
	print("Temperature/(1+z): {:0.2f}+{:0.2f}-{:0.2f} [K]".format(*T_val))
	b_val = res.par_cen('beta')
	print("Beta: {:0.2f}+{:0.2f}-{:0.2f}".format(*b_val))
	res.compute_peaklambda()
	p_val = res.peaklambda_cen()
	print("Peak Obs wavelength: {:0.1f}+{:0.1f}-{0.1f} [um]".format(*p_val))

Note that all the blackbody parameters (temperature, etc.) are 
in the observer frame.  This package is primarily developed in python 3,
but should be compatable with python 2.7.

###Dependencies
This depends on a number of python packages:
* [numpy](http://numpy.scipy.org/)
* [scipy](http://numpy.scipy.org/)
* [astropy](http://www.astropy.org/)
* [cython](http://cython.org/)
* [h5py](http://www.h5py.org/)
* And, perhaps most importantly, [emcee](http://dan.iel.fm/emcee/)

### References
* The affine invariant MCMC code is described in
  [Foreman-Mackey et al. (2012)](http://http://arxiv.org/abs/1202.3665)
* This code has been used in a variety of HerMES papers analyzing
  the SEDs of dusty, star-forming galaxies:
  * [Riechers et al. (2013)](http://dx.doi.org/10.1038/nature12050)
  * [Fu et al. (2013)](http://arxiv.org/abs/1305.4930)
