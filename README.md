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

This has a large number of options which, for example, allow you to compute
the IR luminosity or dustmass as part of the fit.  Help can be obtained via

	run_mbb_emcee.py --help

Carrying out a fit produces a HDF5 output file containing the results,
which can either be read directly, or read back into a mbb_results
object for analysis.  For example, if the output is saved to
the file test.h5:

	import mbb_emcee
	res = mbb_emcee.mbb_results(h5file="test.h5")
	T_val = res.par_cen('T')
	print("Temperature/(1+z): {:0.2f}+{:0.2f}-{:0.2f} [K]".format(*T_val))
	b_val = res.par_cen('beta')
	print("Beta: {:0.2f}+{:0.2f}-{:0.2f}".format(*b_val))
	res.compute_peaklambda()
	p_val = res.peaklambda_cen()
	print("Peak Obs wavelength: {:0.1f}+{:0.1f}-{:0.1f} [um]".format(*p_val))

Note that all the blackbody parameters (temperature, etc.) are 
in the observer frame.  

This package is primarily developed in python 3, but should be
compatable with python 2.7.

###Input file specification
There are two possible input file specifications.  The first treats
all the instrument responses as delta functions, the second integrates
them properly.  The latter is slower, and requires that you know the
appropriate instrument information.

When using the simple delta function approach, the input file specification
are lines of the form: wavelength flux_density flux_density_uncertainty,
where the wavelength is specified in microns, and the flux_density and
it's uncertainty in mJy.  Note that the flux density uncertainties 
in this file are ignored if the user provides a covariance matrix.

In the second the format is similar, except that the wavlength is
replaced with the name of the passband (e.g., SPIRE_250um).  The
information specifying how the name is used is provided in a filter
wheel file.  The user can specify their own such file (in a somewhat
complicated format -- you will need to read the documentation of the
response class), but a default version is also included that should cover
most of the standard passbands.  

However, this doesn't really work for interferometric observations,
where the passbands can usually be tuned in all sorts of complicated
ways.  One way to handle these is to build your own filter wheel file,
but as an alternative a short specification can be provided as the
filter name of the form box_cent_width, where cent and width are the
central frequency and width of a boxcar representation of the filter
in GHz.  So, for example, box_342_8 is a 8 GHz wide boxcar filter
centered at 342 GHz.  Delta function passbands are also allowed using
delta_cent, where, again, cent is the central frequency.  Finally, ALMA
passbands are allowed using alma_cent where, again, cent is the central
frequency in GHz.  Only bands 3, 6, and 7 are currently supported, and
the assumption is that the standard IF ranges are used.  For bands 3 and 7
this means 3.75 GHz coverage, a 8 GHz gap, then 3.75 GHz coverage, with
the center of the 8 GHz gap at the central frequency.  For band 6,
the central gap is 12 GHz.  More complex tunings can't be specified with
the short specifications.

The currently supported passbands in the default set are MIPS_24um,
MIPS_70um, MIPS_160um, PACS_70um, PACS_100um, PACS_160um, SPIRE_250um,
SPIRE_350um, SPIRE_500um, SABOCA_350um, LABOCA_870um, SCUBA2_450um,
SCUBA2_850um, Bolocam_1.1mm, MAMBO2_1.1mm, and GISMO_2mm.

###Examining your results
You should almost always look at both your actual fit and at
histograms of the parameters to make sure there are not long tails of
unphysical values, which can happen when your data is not very
constraining or some parameter.  In particular, be aware of very large
values of lambda0, since that parameter is frequently rather poorly
constrained by sub-mm data, and which will cause the dustmass
computation to produce rather extreme values.  This can be done using
matplotlib via something like:

    import mbb_emcee
    import matplotlib.pyplot as plt
    import numpy as np
    res = mbb_emcee.mbb_results(h5file="test.h5")
    wave, flux, flux_unc = res.data
    p_data = plt.errorbar(wave, flux, yerr=flux_unc, fmt='ro')
    p_wave = np.arange(wave.min() * 0.5, wave.max() * 1.5, 200)
    p_fit = plt.plot(p_wave, res.best_fit_sed(p_wave), color='blue')
    plt.show()
    h = plt.hist(res.parameter_chain('T/(1+z)'))
    plt.show()
    h = plt.hist(res.parameter_chain('beta'))
    plt.show()
    h = plt.hist(res.lir_chain)
    plt.show()

where the latter (plotting the chain of IR luminosities) requires that
you have actually computed them as part of the fit.
    
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
