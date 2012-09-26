import numpy
import math
from modified_blackbody import modified_blackbody

__all__ = ["likelihood"]

"""Class holding data, defining likelihood"""
class likelihood(object) :
    def __init__(self, photfile, covfile=None, covextn=0, 
                 wavenorm=500.0, redshift=None, noalpha=False,
                 opthin=False) :
        """Photfile is the name of the photometry file, covfile the name
        of a fits file holding the covariance matrix (if present), which
        is in extension covextn.  The wavelength normalizing the SED is
        wavenorm (in microns)."""
        self.read_phot( photfile )
        self._z = redshift
        self._wavenorm = wavenorm
        self._noalpha = noalpha
        self._opthin = opthin
        if not covfile is None :
            self.read_cov( covfile, extn=covextn )
        else:
            self._has_covmatrix = False

        # Set up information about fixed params, param limits, and
        # priors.
        # Params are in the order T, beta, lambda0, alpha, fnorm
        self._fixed = [False, False, False, False, False]
        # All parameters have lower limits
        self._lowlim = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        #Setup upper limits
        self._has_uplim = [False, False, False, False, False]
        inf = float("inf")
        self._uplim = np.array([inf, inf, inf, inf, inf])

        #Setup Gaussian prior
        self._any_gprior = False
        self._has_gprior = [False, False, False, False, False]
        self._gprior_mean = np.zeros(5)
        self._gprior_ivar = np.ones(5)

        #Useful for log like with bad params
        self._neginf = float("-inf")

    def read_phot( self, filename ) :
        """Reads in the photometry file, storing the wave [um],
        flux [mJy] and uncertainties [mJy]"""
        import asciitable
        data = asciitable.read(filename,comment='^#')
        if len(data) == 0 :
            errstr = "No data read from %s" % filename
            raise IOError(errstr)
        self._wave     = numpy.array([dat[0] for dat in data])
        self._flux     = numpy.array([dat[1] for dat in data])
        self._flux_unc = numpy.array([dat[2] for dat in data])
        self._ivar     = 1.0/self._flux_unc**2

    def read_cov( self, filename, extn=0 ) :
        """Reads in the covariance matrix from the specified
        extension of the input FITS file (in extension extn)"""
        import pyfits
        import numpy.linalg
        hdu = pyfits.open(filename)
        self._covmatrix = hdu[extn].data
        if self._covmatrix.shape[0] != self._covmatrix.shape[1] :
            errstr = "Covariance matrix from %s is not square" % filename
            raise ValueError(errstr)
        if self._covmatrix.shape[0] != len(self._flux) :
            errstr = "Covariance matrix doesn't have same number of "+\
                "datapoints as photometry"
            raise ValueError(errstr)
        self._invcovmatrix = numpy.linalg.inv(self._covmatrix)
        self._has_covmatrix = True

    def fix_param(self, paramidx):
        """Fixes the specified parameter.

        Params are in order 'T','beta','lambda0','alpha','fnorm'
        """
        self._fixed[paramidx] = True

    def unfix_param(self, paramidx):
        """Un-fixes the specified parameter.

        Params are in order 'T','beta','lambda0','alpha','fnorm'
        """
        self._fixed[paramidx] = False

    def set_lowlim(self, paramidx, val) :
        """Sets the specified parameter lower limit to value.

        Params are in order 'T','beta','lambda0','alpha','fnorm'
        """
        self._lowlim[paramidx] = val

    def set_uplim(self, paramidx, val) :
        """Sets the specified parameter upper limit to value.

        Params are in order 'T','beta','lambda0','alpha','fnorm'
        """
        self._has_uplim[paramidx] = True
        self._uplim[paramidx] = val

    def set_gaussian_prior(self, paramidx, mean, sigma):
        """Sets up a Gaussian prior on the specified parameter.

        Params are in order 'T','beta','lambda0','alpha','fnorm'
        """

        self._any_gprior = True
        self._has_gprior[paramidx] = True
        self._gprior_mean = float(mean)
        self._gprior_ivar = 1.0 / (float(sigma)**2)


    def checklim(self,pars) :
        """Checks to see if a given parameter set passes the limits.
        Returns True if it passes, False if it doesn't"""

        if len(pars) != 5:
            raise ValueError("pars is not of expected length 5")

        for idx, val in enumerate(pars):
            if self._fixed[idx] and val <= self._lowlim[idx]:
                return False
            if self._has_uplim[idx] and (not self._fixed[idx]) and\
                    val > self._uplim[idx]: return False

        return True

    def getsed(self, pars, wave) :
        """Returns the model SED at the specified wavelength -- which
        can be an array (numpy or otherwise)

        The order of pars is T, beta, lambda0, alpha, fnorm
        """

        if len(pars) != 5:
            raise ValueError("pars is not of expected length 5")

        bkb = modified_blackbody(pars[0], pars[1], pars[2], pars[3], pars[4],
                                 wavenorm=self._wavenorm, noalpha=self._noalpha,
                                 opthin=self._opthin)
        return bkb(wave)

    def __call__(self, pars) :
        """Gets log likelihood assuming Gaussian errors: log P( pars | data )

        The order of pars is T, beta, lambda0, alpha, fnorm
        """
        
        #First check limits
        if not self.checklim(pars): return self._neginf

        diff = self._flux - self.getsed(pars,self._wave)
        if self.has_covmatrix:
            lnlike = -0.5*numpy.dot(diff,numpy.dot(self._invcovmatrix,diff))
        else:
            lnlike = -0.5*numpy.sum(diff**2*self._ivar)

        #Add priors
        if self._any_gprior:
            for idx, val in enumerate(pars):
                if self._has_gprior[idx]:
                    delta = val - self._gprior_mean[idx]
                    lnlike -= 0.5 * self._gprior_ivar[idx] * delta**2
        
        return lnlike

    def freqint(self, chain, minfreq, maxfreq, step=0.1) :
        """Integrates greybody between minfreq/maxfreq (in GHz)
        in erg/s/cm^2 (assuming fnorm is mJy), taking an array of parameters
        in the order T, beta, alpha, lambda0, fnorm"""
        nchain = chain.shape[0]
        integral = numpy.zeros(nchain)
        freq = numpy.arange(minfreq,maxfreq,step)
        waveln = 299792458e-3/freq
        ifac = step*1e-17 
        for i in xrange(nchain) :
            pars = chain[i,:]
            model = self.getsed(pars, waveln)
            integral[i] = numpy.sum(model) * ifac
        return integral
    
