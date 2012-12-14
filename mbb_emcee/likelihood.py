import numpy as np
import math
from modified_blackbody import modified_blackbody
import astropy.cosmology

__all__ = ["likelihood"]

""" Parameter order dictionary"""
param_order = {'t': 0, 't/(1+z)': 0, 'beta': 1, 'lambda0': 2,
                'lambda0*(1+z)': 2, 'lambda_0': 2, 'lambda_0*(1+z)': 2,
                'alpha': 3, 'fnorm': 4}

"""Class holding data, defining likelihood"""
class likelihood(object) :

    """ Parameter order dictionary.  Lowercased."""
    _param_order = {'t': 0, 't/(1+z)': 0, 'beta': 1, 'lambda0': 2,
                    'lambda0*(1+z)': 2, 'lambda_0': 2, 'lambda_0*(1+z)': 2,
                    'alpha': 3, 'fnorm': 4}

    def __init__(self, photfile=None, covfile=None, covextn=0, 
                 wavenorm=500.0, noalpha=False, opthin=False) :
        """ Object for computing likelihood of a given set of parameters.

        Parameters
        ----------
        photfile : string
           Text file containing photometry
        
        covfile : string
           FITS file containing covariance matrix. None for no file

        covextn : integer
           Extension of covaraince file

        wavenorm : float
           Wavelength of normalization in microns

        noalpha : bool
           Ignore alpha in fit

        opthin : bool
           Assume optically thin
        """

        self._wavenorm = float(wavenorm)
        self._noalpha = bool(noalpha)
        self._opthin = bool(opthin)

        # Set up information about fixed params, param limits, and
        # priors.
        # Params are in the order T, beta, lambda0, alpha, fnorm
        self._fixed = [False, False, False, False, False]
        # All parameters have lower limits
        # Make these small but slightly larger than 0, except
        # for T, which shouldn't get too small or odd things happen
        # Keep in mind the minimum Temperature should be Tcmb*(1+z)
        # so as long as we stay above that
        self._lowlim = np.array([1, 0.1, 1, 0.1, 1e-3])

        # Setup upper limits; note that alpha and beta have
        # upper limits by default
        self._has_uplim = [False, True, False, True, False]
        inf = float("inf")
        self._uplim = np.array([inf, 20.0, inf, 20.0, inf])

        # Setup Gaussian prior
        self._any_gprior = False
        self._has_gprior = [False, False, False, False, False]
        self._gprior_mean = np.zeros(5)
        self._gprior_ivar = np.ones(5)

        # Data
        if not photfile is None:
            self.read_phot(photfile)
            if not covfile is None :
                if not isinstance(covfile, basestring):
                    raise TypeError("covfile must be string-like")
                self.read_cov(covfile, extn=covextn)
        else:
            if not covfile is None:
                raise Exception("Can't pass in covfile if no photfile")
            self._data_read = False
            self._has_covmatrix = False

        # Reset normalization flux lower limit based on data
        self._lowlim[4] = 1e-3 * self._flux.min()

        self._badval = float("-inf")

    def set_phot(self, wave, flux, flux_unc):
        """ Sets photometry"""
        self._wave = np.asarray(wave)
        self._ndata = len(self._wave) 
        if self._ndata == 0:
            raise ValueError("No elements in wavelength vector")

        self._flux = np.asarray(flux)
        self._flux_unc = np.asarray(flux_unc)
        if len(self._wave) != len(self._flux):
            raise ValueError("wave not same length as flux")
        if len(self._wave) != len(self._flux_unc):
            raise ValueError("wave not same length as flux_unc")
        self._ivar = 1.0 / self._flux_unc**2

        # Set upper limit on lambda0 -- if its 5x above
        #  our longest wavelength point, we can't say anything
        #  about it.
        if not self._has_uplim[2]:
            self._has_uplim[2] = True
            self._uplim[2] = 5.0 * self._wave.max()

        self._data_read = True

    def read_phot(self, filename) :
        """Reads in the photometry file, storing the wave [um],
        flux [mJy] and uncertainties [mJy]"""
        import asciitable
        if not isinstance(filename, basestring):
            raise TypeError("filename must be string-like")
        data = asciitable.read(filename,comment='^#')
        if len(data) == 0 :
            errstr = "No data read from %s" % filename
            raise IOError(errstr)
        self.set_phot([dat[0] for dat in data],[dat[1] for dat in data],
                      [dat[2] for dat in data])

    @property
    def data_read(self):
        return self._data_read

    @property
    def ndata(self):
        if self._data_read:
            return self._ndata
        else:
            return None

    @property 
    def data_wave(self):
        if self._data_read:
            return self._wave
        else:
            return None

    @property 
    def data_flux(self):
        if self._data_read:
            return self._flux
        else:
            return None

    @property 
    def data_flux_unc(self):
        if self._data_read:
            return self._flux_unc
        else:
            return None

    def set_cov(self, covmatrix):
        """ Sets covariance matrix"""

        if not self._data_read:
            raise Exception("Can't set covariance matrix without photometry")

        if covmatrix.shape[0] != covmatrix.shape[1] :
            raise ValueError("Covariance matrix from is not square")
        
        if covmatrix.shape[0] != len(self._flux) :
            errstr = "Covariance matrix doesn't have same number of "+\
                "datapoints as photometry"
            raise ValueError(errstr)

        self._covmatrix = covmatrix
        self._invcovmatrix = np.linalg.inv(self._covmatrix)
        self._has_covmatrix = True


    def read_cov(self, filename, extn=0) :
        """Reads in the covariance matrix from the specified
        extension of the input FITS file (in extension extn)"""
        import pyfits

        if not self._data_read:
            raise Exception("Can't read in covaraince matrix without phot")

        hdu = pyfits.open(filename)
        self.set_cov(hdu[extn].data)

    @property
    def has_data_covmatrix(self):
        return self._has_covmatrix

    @property
    def data_covmatrix(self):
        if self._has_covmatrix:
            return self._covmatrix
        else:
            return None

    def fix_param(self, param):
        """Fixes the specified parameter.

        Parameters
        ----------

        param : int or string
          Parameter specification
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
          Parameter specification
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
          Parameter specification
        """
        if isinstance(param, str):
            self._lowlim[self._param_order[param]] = val
        else:
            self._lowlim[param] = val

    def get_lowlim(self, param):
        """Gets the specified parameter lower limit

        Parameters
        ----------

        param : int or string
          Parameter specification
        """
        if isinstance(param, str):
            paramidx = self._param_order[param.lower()]
        else:
            paramidx = int(param)
            
        return self._lowlim[paramidx]

    def get_lowlims(self):
        return self._lowlim

    def set_uplim(self, param, val) :
        """Sets the specified parameter upper limit to value.

        Parameters
        ----------

        param : int or string
          Parameter specification
        """
        if isinstance(param, str):
            paramidx = self._param_order[param.lower()]
        else:
            paramidx = int(param)

        self._has_uplim[paramidx] = True
        self._uplim[paramidx] = val

    def set_gaussian_prior(self, param, mean, sigma):
        """Sets up a Gaussian prior on the specified parameter.

        Parameters
        ----------

        param : int or string
          Parameter specification
        """

        if isinstance(param, str):
            paramidx = self._param_order[param.lower()]
        else:
            paramidx = int(param)

        self._any_gprior = True
        self._has_gprior[paramidx] = True
        self._gprior_mean[paramidx] = float(mean)
        self._gprior_ivar[paramidx] = 1.0 / (float(sigma)**2)


    def _check_lowlim(self,pars) :
        """Checks to see if a given parameter set passes the lower limits.

        Returns True if it passes, False if it doesn't.

        Unlike the upper limits, in the most common case the
        SED simply can't be compuated below the lower limits,
        so we can't just apply a likelihood penalty.
        """

        if len(pars) != 5:
            raise ValueError("pars is not of expected length 5")

        for idx, val in enumerate(pars):
            if not self._fixed[idx]:
                if val < self._lowlim[idx]:
                    return False

        return True

    def _uplim_prior(self,pars):
        """ Gets likelihood of upper limit priors

        For values above the upper limit, applies a Gaussian
        penalty centered at the limit with sigma = limit/100.0.
        A soft upper limit seems to work better than a hard one"""

        if len(pars) != 5:
            raise ValueError("pars is not of expected length 5")

        logpenalty = 0.0
        for idx, val in enumerate(pars):
            if not self._fixed[idx] and self._has_uplim[idx]:
                lim = self._uplim[idx]
                if val > lim:
                    limvar = (1e-2*lim)**2
                    logpenalty -= 0.5*(val - lim)**2 / limvar

        return logpenalty

    def _set_sed(self, pars):
        """Set up the SED for the provided parameters

        The order of pars is T, beta, lambda0, alpha, fnorm
        """

        if len(pars) != 5:
            raise ValueError("pars is not of expected length 5")
        self._sed = modified_blackbody(pars[0], pars[1], pars[2], pars[3], 
                                       pars[4], wavenorm=self._wavenorm, 
                                       noalpha=self._noalpha,
                                       opthin=self._opthin)
        

    def get_sed(self, pars, wave) :
        """Returns the model SED at the specified wavelength -- which
        can be an array (numpy or otherwise)

        The order of pars is T, beta, lambda0, alpha, fnorm
        """

        self._set_sed(pars)
        return self._sed(wave)

    def __call__(self, pars) :
        """Gets log likelihood assuming Gaussian errors: log P( pars | data )

        The order of pars is T, beta, lambda0, alpha, fnorm
        """
        
        # First check limits
        # Return large negative number if bad
        if not self._check_lowlim(pars): return self._badval

        # Set params in model
        self._set_sed(pars)

        # Assume Gaussian uncertanties, ignore constant prefactor
        diff = self._flux - self._sed(self._wave)
        if self._has_covmatrix:
            lnlike = -0.5*np.dot(diff,np.dot(self._invcovmatrix,diff))
        else:
            lnlike = -0.5*np.sum(diff**2*self._ivar)

        # Add in upper limit priors
        lnlike += self._uplim_prior(pars)

        # Add Gaussian priors
        if self._any_gprior:
            for idx, val in enumerate(pars):
                if self._has_gprior[idx]:
                    delta = val - self._gprior_mean[idx]
                    lnlike -= 0.5 * self._gprior_ivar[idx] * delta**2

        return lnlike

