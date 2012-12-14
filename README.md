mbb_emcee
========================

This is a package to fit modified blackbodies to photometry
data using an affine invariant MCMC.

###Installation
The usual
	python setup.py install

###Usage
The command line routine is
	run_mbb_emcee.py

Help can be obtained via
	mbb_emcee.py --help

This produces a pickled save file containing the results.
You can obtain parameter ranges and limits by reading
in the results and examining them.  Note that all the
blackbody parameters (temperature, etc.) are in the observer frame.
If picklefile is a string holding the name of the output file:
	import pickle
	results = pickle.load(open(picklefile,'rb'))
        T_val = results.par_cen(0)
	print "Obs frame Temperature: {:0.2f}+{:0.2f}-{:0.2f}".format(*T_val)
	b_val = results.par_cen(1)
	print "Beta: {:0.2f}+{:0.2f}-{:0.2f}".format(*b_val)

###Dependencies
This depends on a number of python packages:
* [numpy](http://http://numpy.scipy.org/)
* [scipy](http://http://numpy.scipy.org/)
* [pyfits](http://http://www.stsci.edu/institute/software_hardware/pyfits)
* And, perhaps most importantly, [emcee](http://http://danfm.ca/emcee/)
* If you want to compute dust masses, you will also need
  [astropy](http://www.astropy.org/)

### References
* The affine invariant MCMC code is described in
  [Foreman-Mackey et al. (2012)](http://http://arxiv.org/abs/1202.3665)
