import numpy as np
import math
from modified_blackbody import modified_blackbody
from response import response, response_set
import astropy.cosmology

__all__ = ["likelihood"]

"""Class holding data, defining likelihood"""
class likelihood(object) :

    """ Parameter order dictionary.  Lowercased."""
    _param_order = {'t': 0, 't/(1+z)': 0, 'beta': 1, 'lambda0': 2,
                    'lambda0*(1+z)': 2, 'lambda_0': 2, 'lambda_0*(1+z)': 2,
                    'alpha': 3, 'fnorm': 4}

    def __init__(self, photfile=None, covfile=None, covextn=0, 
                 wavenorm=500.0, noalpha=False, opthin=False,
                 responsefile=None, responsedir=None) :
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

        responsefile : string
           Name of file containing response specifications.  If set,
           response integration is used in the fitting.

        responsedir : string
           Directory to look for response files in.
        """

        self._wavenorm = float(wavenorm)
        self._noalpha = bool(noalpha)
        self._opthin = bool(opthin)

        # Set up information about fixed params, param limits, and
        # priors.

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
        self._gprior_sigma = np.zeros(5)
        self._gprior_ivar = np.ones(5)

        # Responses
        self._response_integrate = False
        if not responsefile is None:
            self.read_responses(responsefile, responsedir=responsedir)

        # Data
        self._data_read = False
        self._has_covmatrix = False
        if not photfile is None:
            self.read_phot(photfile)
            if not covfile is None :
                if not isinstance(covfile, basestring):
                    raise TypeError("covfile must be string-like")
                self.read_cov(covfile, extn=covextn)
        else:
            if not covfile is None:
                raise Exception("Can't pass in covfile if no photfile")

        # Reset normalization flux lower limit based on data
        self._lowlim[4] = 1e-3 * self._flux.min()

        self._badval = float("-inf")

    @property
    def wavenorm(self):
        """ Normalization wavelength in microns"""
        return self._wavenorm

    @property
    def noalpha(self):
        """ Not including a blue side power law?"""
        return self._noalpha

    @property
    def opthin(self):
        """ Assuming an optically thin model?"""
        return self._opthin

    @property
    def response_integrate(self):
        """Is filter integration being used?"""
        return self._response_integrate
        
    def read_responses(self, responsefile, responsedir=None):
        """ Read in responses

        Parameters
        ----------
        responsefile : string
          File containing filter specification information.

        responsedir : string
          Directory to look for actual responses in.
          
        Notes
        -----
        Calling this turns on filter integration
        """
        self._responsewheel = response_set(responsefile, dir=responsedir)
        self._response_integrate = True


    def set_phot(self, firstarg, flux, flux_unc):
        """ Sets photometry

        Parameters
        ----------

        firstarg : array like
          If using response integration, a string array of response names.
          Otherwise, an array of wavelengths, in microns

        flux : array like
          Array of flux densities, in mJy

        flux_unc : array like
          Array of flux density uncertainties, in mJy
        
        Notes
        -----
        This wipes out any covariance matrix already present,
        and turns off response integration
        """

        if self._response_integrate:
            # Get filter responses in same order as photometry
            if not isinstance(firstarg[0], basestring):
                raise ValueError("Expecting response string name")
            self._responses = []
            for name in firstarg:
                # Do it this way to provide a more helpful error message
                # if name is not known
                if not self._responsewheel.has_key(name):
                    raise ValueError("Unknown filter response %s" % name)
                self._responses.append(self._responsewheel[name])

            self._response_names = [r.name for r in self._responses]
            self._wave = [resp.effective_wavelength for resp in self._responses]
        else:
            self._wave = np.asarray(firstarg, dtype=np.float64)
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
        self._has_covmatrix = False


    def read_phot(self, filename):
        """Reads in the photometry file

        Parameters
        ----------
        filename : string
          Name of file to read input from.  This file should have
          three columns: the wavelength [microns], the flux density
          [mJy], and the uncertainty in the flux density [mJy].

        Notes
        -----
        This wipes out any covariance matrix present
        """

        import astropy.io.ascii

        if not isinstance(filename, basestring):
            raise TypeError("filename must be string-like")

        data = astropy.io.ascii.read(filename,comment='^#')

        if len(data) == 0 :
            errstr = "No data read from %s" % filename
            raise IOError(errstr)

        self.set_phot([dat[0] for dat in data],
                      [dat[1] for dat in data],
                      [dat[2] for dat in data])

    @property
    def data_read(self):
        """ Has the data been set?"""
        return self._data_read

    @property
    def ndata(self):
        """ The number of data points"""
        if self._data_read:
            return self._ndata
        else:
            return 0

    @property 
    def data_wave(self):
        """ The data wavelengths, in microns"""
        if self._data_read:
            return self._wave
        else:
            return None

    @property 
    def response_names(self):
        if not hasattr(self, '_response_names'): return None
        return self._response_names
        
    def get_response(self, name):
        """ Return the matching response object"""
        if not hasattr(self, '_responsewheel'): return None
        return self._responsewheel[name]
        
    @property 
    def data_flux(self):
        """ The flux densities, in mJy"""
        if self._data_read:
            return self._flux
        else:
            return None

    @property 
    def data_flux_unc(self):
        """ The uncertainties in the flux densities, in mJy.

        Notes
        -----
        If a covariance matrix is available, these values are
        not used by the fits.
        """

        if self._data_read:
            if self._has_covmatrix:
                return np.sqrt(np.diag(self._covmatrix))
            else:
                return self._flux_unc
        else:
            return None

    def set_cov(self, covmatrix):
        """ Sets covariance matrix

        Parameters
        ----------

        covmatrix : array like
          Covariance matrix
        """

        if not self._data_read:
            raise Exception("Can't set covariance matrix without photometry")

        if len(covmatrix.shape) != 2:
            raise ValueError("Covariance matrix is not 2 dimensional")

        if covmatrix.shape[0] != covmatrix.shape[1]:
            raise ValueError("Covariance matrix from is not square")
        
        if covmatrix.shape[0] != len(self._flux):
            errstr = "Covariance matrix doesn't have same number of "+\
                "datapoints as photometry"
            raise ValueError(errstr)

        self._covmatrix = covmatrix
        self._invcovmatrix = np.linalg.inv(self._covmatrix)
        self._has_covmatrix = True


    def read_cov(self, filename, extn=0) :
        """Reads in the covariance matrix from a FITS file.
        
        Parameters
        ----------
        filename : string
          File to read covariance matrix from
        
        extn : int
          Extension to look for covariance matrix in
        """
        
        import astropy.io.fits

        if not self._data_read:
            raise Exception("Can't read in covaraince matrix without phot")
        hdu = astropy.io.fits.open(filename)
        self.set_cov(hdu[extn].data)

    @property
    def has_data_covmatrix(self):
        """ Does this object have a flux density covariance matrix?"""
        return self._has_covmatrix

    @property
    def data_covmatrix(self):
        """ The covariance matrix of the flux densities, in mJy^2"""
        if self._has_covmatrix:
            return self._covmatrix
        else:
            return None

    @property
    def data_invcovmatrix(self):
        """ Get inverse covariance matrix"""
        if self._has_covmatrix:
            return self._invcovmatrix
        else:
            return None

    def get_paramindex(self, paramname):
        """ Convert the name of a parameter into its index.

        Parameters
        ----------
        paramname : string
          Name of parameter (e.g., 'beta')
        
        Returns
        -------
        index : int
          Parameter index
        """
        
        return self._param_order[paramname]

    def set_lowlim(self, param, val) :
        """Sets the specified parameter lower limit to value.

        Parameters
        ----------

        param : int or string
          Parameter specification. Either an index into
          the parameter list, or a string name for the 
          parameter.
        """

        if isinstance(param, str):
            self._lowlim[self._param_order[param.lower()]] = val
        else:
            self._lowlim[param] = val

    def lowlim(self, param):
        """Gets the specified parameter lower limit

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
            
        return self._lowlim[paramidx]

    @property
    def lowlims(self):
        """ Get the list of lower parameter limits.

        Returns
        -------
        lowlims : ndarray
          The order is T/(1+z), beta, lambda0 (1+z), alpha, fnorm.
        """
        return self._lowlim

    def set_uplim(self, param, val) :
        """Sets the specified parameter upper limit to value.

        Parameters
        ----------

        param : int or string
          Parameter specification. Either an index into
          the parameter list, or a string name for the 
          parameter.
        
        val : float
          The upper limit to set.
        """

        if isinstance(param, str):
            paramidx = self._param_order[param.lower()]
        else:
            paramidx = int(param)

        self._has_uplim[paramidx] = True
        self._uplim[paramidx] = val

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
        if isinstance(param, str):
            paramidx = self._param_order[param.lower()]
        else:
            paramidx = int(param)
        
        return self._has_uplim[paramidx]

    def uplim(self, param):
        """ What is the upper limit for a given parameter?

        Parameters
        ----------

        param : int or string
          Parameter specification.  Either an index into
          the parameter list, or a string name for the 
          parameter.

        Returns
        -------
        val : float
          Upper limit, or None if there isn't one
        """
        if isinstance(param, str):
            paramidx = self._param_order[param.lower()]
        else:
            paramidx = int(param)
        
        if not self._has_uplim[paramidx]: return None
        return self._uplim[paramidx]


    def set_gaussian_prior(self, param, mean, sigma):
        """Sets up a Gaussian prior on the specified parameter.

        Parameters
        ----------

        param : int or string
          Parameter specification.  Either an index into
          the parameter list, or a string name for the 
          parameter.

        mean : float
          Mean of Gaussian prior

        sigma : float
          Sigma of Gaussian prior
        """

        if isinstance(param, str):
            paramidx = self._param_order[param.lower()]
        else:
            paramidx = int(param)

        self._any_gprior = True
        self._has_gprior[paramidx] = True
        self._gprior_mean[paramidx] = float(mean)
        self._gprior_sigma = float(sigma)
        self._gprior_ivar[paramidx] = 1.0 / (float(sigma)**2)

    def has_gaussian_prior(self, param):
        """ Does the given parameter have a Gaussian prior set?

        Parameters
        ----------

        param : int or string
          Parameter specification.  Either an index into
          the parameter list, or a string name for the 
          parameter.
          
        Returns
        -------
        has_prior : bool
          True if a Gaussian prior is set, False otherwise
        """
        
        if isinstance(param, str):
            paramidx = self._param_order[param.lower()]
        else:
            paramidx = int(param)

        return self._has_gprior[paramidx]

    def get_gaussian_prior(self, param):
        """ Return Gaussian prior values

        Parameters
        ----------

        param : int or string
          Parameter specification.  Either an index into
          the parameter list, or a string name for the 
          parameter.
          
        Returns
        -------
        tup : tuple or None
          Mean, variance if set, None otherwise
        """
        
        if not self._any_gprior: return None

        if isinstance(param, str):
            paramidx = self._param_order[param.lower()]
        else:
            paramidx = int(param)
            
        if not self._has_gprior[paramidx]: return None
        return (self._gprior_mean[paramidx], self._gprior_sigma[paramidx])


    def _check_lowlim(self,pars) :
        """Checks to see if a given parameter set passes the lower limits.

        Parameters
        ----------
        pars : array like
          5 element list of parameters (T, beta, lambda0, alpha, fnorm).

        Returns
        -------
        pass_check : bool
          True if it passes, False if it doesn't.

        Notes
        -----
        Unlike the upper limits, in the most common case the
        SED simply can't be compuated below the lower limits,
        so we can't just apply a likelihood penalty.
        """

        if len(pars) != 5:
            raise ValueError("pars is not of expected length 5")

        for idx, val in enumerate(pars):
            if val < self._lowlim[idx]:
                return False

        return True

    def _uplim_prior(self,pars):
        """ Gets log likelihood of upper limit priors

        Parameters
        ----------
        pars : array like
          5 element list of parameters (T, beta, lambda0, alpha, fnorm).

        Returns
        -------
        like_penalty : float
         Penalty to apply to log likelihood based on these parameters.
         This should be added to the log likelihood (that is, it is
         negative).

        Notes
        -----
        For values above the upper limit, applies a Gaussian
        penalty centered at the limit with sigma = limit range/50.0.
        A soft upper limit seems to work better than a hard one.
        """

        if len(pars) != 5:
            raise ValueError("pars is not of expected length 5")

        logpenalty = 0.0
        for idx, val in enumerate(pars):
            if self._has_uplim[idx]:
                lim = self._uplim[idx]
                if val > lim:
                    limvar = (0.02 * (lim - self._lowlim[idx]))**2
                    logpenalty -= 0.5*(val - lim)**2 / limvar

        return logpenalty

    def _set_sed(self, pars):
        """ Set up the SED for the provided parameters

        Parameters
        ----------
        pars : array like
          5 element list of parameters (T, beta, lambda0, alpha, fnorm).
        """

        if len(pars) != 5:
            raise ValueError("pars is not of expected length 5")
        self._sed = modified_blackbody(pars[0], pars[1], pars[2], pars[3], 
                                       pars[4], wavenorm=self._wavenorm, 
                                       noalpha=self._noalpha,
                                       opthin=self._opthin)
        

    def get_sed(self, pars, wave):
        """ Get the model SED at the specified wavelengths for a set of params.

        Parameters
        ----------
        pars : array like
          5 element list of parameters (T, beta, lambda0, alpha, fnorm).

        wave : array like
          Wavelengths, in microns

        Returns
        -------
        sed : ndarray
          SED of the parameters in mJy at the specified wavelengths.
        """

        self._set_sed(pars)
        return self._sed(wave)


    def __call__(self, pars) :
        """ Gets log likelihood of the parameters.

        Parameters
        ----------
        pars : array like
          5 element list of parameters (T, beta, lambda0, alpha, fnorm).

        Returns
        -------
        log_likelihood : float
          log P(pars | data), including priors and limits.
        """
        
        # First check limits
        # Return large negative number if bad
        if not self._check_lowlim(pars): return self._badval

        # Set up SED model
        self._set_sed(pars)

        # Get model fluxes for comparison with data
        if self._filter_integrate:
            model_flux = np.array([resp(self._sed) for 
                                   resp in self._responses])
        else:
            model_flux = self._sed(self._wave)

        # Compute likelihood
        #  Assume Gaussian uncertanties, ignore constant prefactor
        diff = self._flux - model_flux
        if self._has_covmatrix:
            lnlike = -0.5 * np.dot(diff, np.dot(self._invcovmatrix, diff))
        else:
            lnlike = -0.5 * np.sum(diff**2 * self._ivar)

        # Add in upper limit priors to likelihood
        lnlike += self._uplim_prior(pars)

        # Add Gaussian priors to likelihood
        if self._any_gprior:
            for idx, val in enumerate(pars):
                if self._has_gprior[idx]:
                    delta = val - self._gprior_mean[idx]
                    lnlike -= 0.5 * self._gprior_ivar[idx] * delta**2

        return lnlike

