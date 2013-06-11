from __future__ import print_function

import numpy as np
import emcee
import math
import multiprocessing
from .modified_blackbody import modified_blackbody
from .likelihood import likelihood

__all__ = ["mbb_fitter"]

class mbb_fitter(object):
    """ Does fit"""

    """ Parameter order dictionary.  Lowercased."""
    _param_order = {'t': 0, 't/(1+z)': 0, 'beta': 1, 'lambda0': 2,
                    'lambda0*(1+z)': 2, 'lambda_0': 2, 'lambda_0*(1+z)': 2,
                    'alpha': 3, 'fnorm': 4, 'f500': 4, 'lambda_peak': 5,
                    'peaklam': 5}

    """ Parameter names"""
    _parnames = np.array(['T/(1+z)', 'Beta', 'Lambda0*(1+z)', 
                          'Alpha', 'Fnorm'])


    def __init__(self, nwalkers=250, photfile=None, covfile=None, 
                 covextn=0, responsefile=None, responsedir=None,
                 wavenorm=500.0, noalpha=False, 
                 opthin=False, nthreads=1):
        """
        Parameters
        ----------

        nwalkers : integer
           Number of MCMC walkers to use in fit

        photfile : string
           Text file containing photometry
        
        covfile : string
           FITS file containing covariance matrix. None for no file

        covextn : integer
           Extension of covaraince file

        responsefile : string
           Name of response specification file

        responsedir : string
           Directory to look for response information in

        wavenorm : float
           Wavelength of normalization in microns

        noalpha : bool
           Ignore alpha in fit

        opthin : bool
           Assume optically thin

        nthreads : integer
           Number of threads to use
        """

        self._noalpha = noalpha
        self._opthin = opthin
        self._wavenorm = float(wavenorm)
        self._nwalkers = int(nwalkers)
        self._nthreads = int(nthreads)
        self.like = likelihood(photfile=photfile, covfile=covfile, 
                               covextn=covextn, wavenorm=wavenorm, 
                               noalpha=noalpha, opthin=opthin,
                               responsefile=responsefile,
                               responsedir=responsedir)
        self.sampler = emcee.EnsembleSampler(self._nwalkers, 5, self.like,
                                             threads=self._nthreads)
        self._sampled = False

        # Params are in the order T, beta, lambda0, alpha, fnorm
        self._fixed = [False, False, False, False, False]

    @property
    def noalpha(self):
        """ Not using blue side power law?"""
        return self._noalpha
    
    @property
    def opthin(self):
        """ Assuming optically thin model?"""
        return self._opthin

    @property
    def wavenorm(self):
        return self._wavenorm

    @property
    def nwalkers(self):
        """ Number of walkers"""
        return self._nwalkers

    @property
    def nthreads(self):
        """ Number of threads"""
        return self._nthreads

    @property
    def sampled(self):
        """ Has the distribution been sampled?"""
        return self._sampled

    @property
    def fixed(self):
        """ Which parameters are fixed?"""
        return self._fixed

    def read_data(self, photfile, covfile=None, covextn=0,
                  responsefile=None, responsedir=None):
        """ Read in photometry data from files

        Parameters
        ----------
        photfile : string
           Text file containing photometry
        
        covfile : string
           FITS file containing covariance matrix. None for no file

        covextn : integer
           Extension of covaraince file

        responsefile : string
           Name of response specification file

        responsedir : string
           Directory to look for responses in.

        Notes
        -----
        Setting responsefile will result in filter integration being
        turned on for the fit.
        """

        if not responsefile is None:
            self.like.read_responses(responsefile, 
                                     responsedir=responsedir)

        self.like.read_phot(photfile)
        if not covfile is None:
            self.like.read_cov(covfile, extn=covextn)

    @property
    def response_integrate(self):
        """Is response integrating in use"""
        return self.like.response_integrate

    def set_data(self, wave, flux, flux_unc, covmatrix=None):
        """ Set photometry and covariance matrix

        Parameters
        ----------
        wave : array like
          Wavelength of data points in microns

        flux : array like
          Flux density of data in mJy

        flux_unc : array like
          Uncertainty in flux density of data in mJy

        covmatrix : array like
          Covariance matrix of data in mJy^2
        """

        self.like.set_phot(wave, flux, flux_unc)
        if not covmatrix is None:
            self.like.set_cov(covmatrix)

    def fix_param(self, param):
        """Fixes the specified parameter.

        Parameters
        ----------

        param : int or string
          Parameter specification.  Either an index into
          the parameter list, or a string name for the 
          parameter.
        """

        if isinstance(param, str):
            paramidx = self._param_order[param.lower()]
        else:
            paramidx = int(param)
            
        self._fixed[paramidx] = True

    def unfix_param(self, param):
        """Un-fixes the specified parameter.
        
        Parameters
        ----------

        param : int or string
          Parameter specification. Either an index into
          the parameter list, or a string name for the 
          parameter.
        """

        if isinstance(param, str):
            paramidx = self._param_order[param.lower()]
        else:
            paramidx = int(param)
            
        self._fixed[paramidx] = False

    def set_lowlim(self, param, val) :
        """Sets the specified parameter lower limit to value.

        Parameters
        ----------

        param : int or string
          Parameter specification. Either an index into
          the parameter list, or a string name for the 
          parameter.
        """
        self.like.set_lowlim(param, val)

    def lowlim(self, param) :
        """Gets the specified parameter lower limit value.

        Parameters
        ----------
        param : int or string
          Parameter specification. Either an index into
          the parameter list, or a string name for the 
          parameter.

        Returns 
        -------
        limit : float
          Lower limit on parameter value
        """
        return self.like.lowlim(param)


    def set_uplim(self, param, val) :
        """Sets the specified parameter upper limit to value.

        Parameters
        ----------

        param : int or string
          Parameter specification. Either an index into
          the parameter list, or a string name for the 
          parameter.  The peak lambda in microns (observer frame)
          can also be used as a parameter (parameter 5, or 'lambda_peak')
        """
        self.like.set_uplim(param, val)

    def has_uplim(self, param):
        """ Does the likelihood have an upper limit for a given parameter?

        Parameters
        ----------

        param : int or string
          Parameter specification.  Either an index into
          the parameter list, or a string name for the 
          parameter.

        Returns
        -------
        val : bool
          True if there is an upper limit, false otherwise
        """
        return self.like.has_uplim(param)

    def uplim(self, param) :
        """Gets the specified parameter upper limit value.

        Parameters
        ----------
        param : int or string
          Parameter specification. Either an index into
          the parameter list, or a string name for the 
          parameter.

        Returns 
        -------
        limit : float
          Upper limit on parameter value, or None if there is none
        """
        return self.like.uplim(param)


    def set_gaussian_prior(self, param, mean, sigma):
        """Sets up a Gaussian prior on the specified parameter.

        Parameters
        ----------

        param : int or string
          Parameter specification.  Either an index into
          the parameter list, or a string name for the 
          parameter.  The peak lambda in microns (observer 
          frame) can also be used as a parameter (parameter 5, 
          or 'lambda_peak').

        mean : float
          Mean of Gaussian prior

        sigma : float
          Sigma of Gaussian prior
        """
        self.like.set_gaussian_prior(param, mean, sigma)

    def has_gaussian_prior(self, param):
        """ Does the given parameter have a Gaussian prior set?

        Parameters
        ----------

        param : int or string
          Parameter specification.  Either an index into
          the parameter list, or a string name for the 
          parameter.  The peak lambda in microns (observer 
          frame) can also be used as a parameter (parameter 5, 
          or 'lambda_peak').
          
        Returns
        -------
        has_prior : bool
          True if a Gaussian prior is set, False otherwise
        """
        return self.like.has_gaussian_prior(param)

    def get_gaussian_prior(self, param):
        """ Return Gaussian prior values

        Parameters
        ----------

        param : int or string
          Parameter specification.  Either an index into
          the parameter list, or a string name for the 
          parameter.  The peak lambda in microns (observer 
          frame) can also be used as a parameter (parameter 5, 
          or 'lambda_peak')
          
        Returns
        -------
        tup : tuple or None
          Mean, variance if set, None otherwise
        """
        return self.like.get_gaussian_prior(param)

    def generate_initial_values(self, initvals, initsigma):
        """ Generate a set of nvalues initial parameters.

        Parameters
        ----------
        initvals : ndarray
          Initial parameter values in order T, beta, lambda0, alpha, fnorm.

        initsigma : ndarray
          Sigma values for each parameter.  Ignored if parameter is fixed.

        Returns
        -------
        p0 : ndarray
          A nwalkers x 5 array of initial values.

        Notes
        -----
        This is only interesting because it has to obey parameter limits.
        As a consequence, the user-provided initial values may be ignored.
        This will take care of fixed parameters correctly.  This ignores
        the peak lambda upper limit.
        """

        import copy

        if len(initvals) != 5:
            raise ValueError("Initial values not expected length")
        if len(initsigma) != 5:
            raise ValueError("Initial sigma values not expected length")
        
        # First -- see if any of the initial values lie outside
        # the upper/lower limits.
        outside_limits = [False] * 5
        for i, val in enumerate(initvals):
            #Everything has a lower limit...
            if val < self.lowlim(i): 
                outside_limits[i] = True
            elif self.has_uplim(i) and val > self.uplim(i):
                outside_limits[i] = True
                
        # If any of the things outside the limits are fixed, this
        # is a problem.  Complain.
        fixed_and_outside = np.logical_and(self._fixed, outside_limits)
        if fixed_and_outside.any():
            badparams = \
                ', '.join(self._parnames[fixed_and_outside.nonzero()[0]])
            errmsg = "Some fixed parameters outside limits: %s"
            raise ValueError(errmsg % badparams)
                                  
        # Adjust initial values if necessary.  We try to keep them
        # close to the user provided value, just shifting into the
        # range by a few sigma.  If the range is too small, just
        # stick it in the middle.
        # A fixed parameter that is out of range is a problem.
        int_init = np.zeros(5)
        for i in range(5):
            if outside_limits[i]:
                if self.has_uplim(i):
                    par_range = self.uplim(i) - self.lowlim(i)
                    if par_range <= 0:
                        raise ValueError("Limits on parameter %d cross" % i)
                    if 2.0 * initsigma[i] >= par_range:
                        int_init[i] = self.lowlim(i) + 0.5 * par_range
                    else:
                        # Figure out if we are below or above
                        if initvals[i] < self.lowlim(i):
                            int_init[i] = self.lowlim(i) + 2 * initsigma[i]
                        else:
                            int_init[i] = self.uplim(i) - 2 * initsigma[i]
                else:
                    int_init[i] = self.lowlim(i) + 2 * initsigma[i]
            else:
                int_init[i] = initvals[i]
                        
        # Make p0 (output)
        p0 = np.zeros((self._nwalkers, 5))
        maxiters = 100
        for i in range(5):
            if self._fixed[i]:
                p0[:,i] = int_init[i] * np.ones(self._nwalkers)
            else:
                lwlim = self.lowlim(i)
                hs_uplim = self.has_uplim(i)
                uplim = self.uplim(i)

                pvec = initsigma[i] * np.random.randn(self._nwalkers) +\
                    int_init[i]

                # Now reject and regenerate anything outside limits
                if hs_uplim:
                    badidx = np.logical_or(pvec > uplim,
                                           pvec < lwlim).nonzero()[0]
                else:
                    badidx = np.nonzero(pvec < lwlim)[0]
                iters = 0
                nbad = len(badidx)
                while nbad > 0:
                    pvec[badidx] = initsigma[i] * np.random.randn(nbad) + \
                        int_init[i]
                    iters += 1

                    if hs_uplim:
                        badidx = \
                            np.logical_or(pvec > uplim,
                                          pvec < lwlim).nonzero()[0]
                    else:
                        badidx = np.nonzero(pvec < lwlim)[0]
                    nbad = len(badidx)

                    if iters > maxiters:
                        errmsg = "Too many iterations initializing param %d"
                        raise Exception(errmsg % i)

                p0[:,i] = pvec

        return p0
            

    def run(self, nburn, nsteps, p0, verbose=False):
        """Do emcee run.

        Parameters
        ----------
        nburn : int
          Number of burn in steps to do

        nsteps : int
          Number of steps to do for each walker

        p0 : ndarray
          Array of initial positions for each walker,
          dimension nwalkers by 5.

        verbose : bool
          Print out informational messages during run.
        """

        # Make sure we have data
        if not self.like.data_read:
            raise Exception("Data not read, needed to do fit")

        if verbose:
            print("Starting fit")
            if self.response_integrate:
                print("  Using response integration")

        # Make sure initial parameters are valid
        for i in range(5):
            # Don't test parameters we don't use
            if i == 2 and self._opthin: continue
            if i == 3 and self._noalpha: continue
            # Limit check
            if self.has_uplim(i) and p0[:,i].max() > self.uplim(i):
                errmsg = "Upper limit initial value violation for %s"
                raise ValueError(errmsg % self._parnames[i])
            if p0[:,i].min() < self.lowlim(i):
                errmsg = "Lower limit initial value violation for %s"
                raise ValueError(errmsg % self._parnames[i])
                
        # Do burn in
        self.sampler.reset()
        self._sampled = False

        if nburn <= 0:
            errmsg = "Invalid (non-positive) number of burn in steps: %d"
            raise ValueError(errmsg % nburn)
        if verbose:
            print("  Doing burn in with %d steps" % nburn)
        pos, prob, rstate = self.sampler.run_mcmc(p0, nburn)

        # Reset and do main fit
        self.sampler.reset()
        if nsteps <= 0:
            errmsg = "Invalid (non-positive) number of main chain steps: %d"
            raise ValueError(errmsg % nsteps)
        if verbose:
            print("  Doing main chain with %d steps" % nsteps)
        st = self.sampler.run_mcmc(pos, nsteps, rstate0=rstate)
        self._sampled = True

        if verbose:
            print("  Fit complete")
            print("   Mean acceptance fraction:", 
                  np.mean(self.sampler.acceptance_fraction))
            try :
                acor = self.sampler.acor
                print("   Autocorrelation time: ")
                print("    Number of burn in steps (%d) should be larger "
                      "than these" % nburn)
                print("\tT:        %f" % acor[0])
                print("\tbeta:     %f" % acor[1])
                if not self._opthin:
                    print("\tlambda0:  %f" % acor[2])
                if not self._noalpha:
                    print("\talpha:    %f" % acor[3])
                print("\tfnorm:    %f" % acor[4])
            except ImportError :
                pass
