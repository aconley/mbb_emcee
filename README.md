modified_blackbody_emcee
========================

This is a package to fit modified blackbodies to photometry
data using an affine invariant MCMC.

###Usage
The command line routine is
	modified_blackbody_emcee.py

Help can be obtained via
	modified_blackbody_emcee.py --help

###Dependencies
This depends on a number of python packages:
* [numpy](http://http://numpy.scipy.org/)
* [scipy](http://http://numpy.scipy.org/)
* [pyfits](http://http://www.stsci.edu/institute/software_hardware/pyfits)
* And, perhaps most importantly, [emcee](http://http://danfm.ca/emcee/)

### References
* The affine invariant MCMC code is described in
  [Foreman-Mackey et al. (2012)](http://http://arxiv.org/abs/1202.3665)
