from __future__ import print_function

import numpy as np
import emcee
import math
import multiprocessing
import astropy.units as u
import astropy.cosmology
from .modified_blackbody import modified_blackbody
from .response import response_set
from .mbb_fit import mbb_fitter
import copy

__all__ = ["mbb_results"]

#hack for basestring
try:
    basestring
except:
    #Python 3
    basestring = str


############################################################

# This class holds the results.  Unlike mbb_fitter, it can be
# serialized.

class mbb_results(object):
    """Holds results of fit"""

    """ Parameter order dictionary.  Lowercased."""
    _param_order = {'t': 0, 't/(1+z)': 0, 'beta': 1, 'lambda0': 2,
                    'lambda0*(1+z)': 2, 'lambda_0': 2, 'lambda_0*(1+z)': 2,
                    'alpha': 3, 'fnorm': 4, 'f500': 4}

    def __init__(self, fit=None, h5file=None, redshift=None, lumdist=None,
                 cosmo_type='WMAP9'):
        """
        Parameters
        ----------
        fit : mbb_fitter
          Fit object

        h5file : string
          Name of HDF5 file to load from previously saved fit.  You can't
          provide this in combination with any of the other variables.

        redshift : float
          Redshift of source.  Necessary if you plan to compute
          dustmass, L_IR, or L_AGN

        lumdist : float or astropy.units.Quantity
          Luminosity distance to object, in Mpc if units not specified.

        cosmo_type : string
          Name of cosmology to use if needed.  The pre-loaded
          cosmologies from astropy.cosmology are supported (e.g.,
          'WMAP9', 'Planck13').  This is used if lumdist is not provided
        """

        # Make sure user isn't trying to specify too much
        if not h5file is None:
            if not fit is None:
                raise Exception("It doesn't make sense to provide h5file"
                                " and a fit to process")
            if not redshift is None:
                raise Exception("It doesn't make sense to provide h5file"
                                " and the redshift")
            if not lumdist is None:
                raise Exception("It doesn't make sense to provide h5file"
                                " and the lumdist")

        self._fitset = False
        self._has_lir = False
        self._lir_min = None
        self._lir_max = None
        self._has_dustmass = False
        self._kappa = None
        self._kappa_wave = None
        self._has_peaklambda = False

        if not h5file is None:
            self.readFromHDF5(h5file)
        else:
            if redshift is None:
                self._z = None
            else:
                self._z = float(redshift)

            if lumdist is None:
                self._has_lumdist = False
            else:
                self._has_lumdist = True
                if isinstance(lumdist, u.Quantity):
                    self._lumdist = lumdist.to(u.Mpc)
                else:
                    self._lumdist = u.Quantity(float(lumdist), u.Mpc)
            if cosmo_type is None:
                raise ValueError("Cosmology type must not be none -- "
                                 "maybe you should just use the default")
            self._cosmo_type = cosmo_type

            if not fit is None:
                self.process_fit(fit)

    def process_fit(self, fit):
        """ Process the fit"""

        if not isinstance(fit, mbb_fitter):
            raise ValueError("Input is not of type mbb_fit")

        self._fitset = True

        # Fit specification
        self._noalpha = fit.noalpha
        self._opthin = fit.opthin
        self._wavenorm = fit.wavenorm
        self._nwalkers = fit.nwalkers
        self._lowlim = fit.like.lowlims
        self._has_uplim = fit.like.has_uplims
        self._uplim = fit.like.uplims
        self._has_gprior = fit.like.has_gpriors
        self._gprior_mean = fit.like.gprior_means
        self._gprior_sigma = fit.like.gprior_sigmas
        self._gprior_ivar = fit.like.gprior_ivars
        self._fixed = fit._fixed
        self._ndata = fit.like.ndata

        # Filter responses
        self._response_integrate = fit.like.response_integrate
        if self._response_integrate:
            self._responsewheel = fit.like._responsewheel
        elif hasattr(self, '_responsewheel'):
            del self._responsewheel

        # Data
        self._data_wave = fit.like.data_wave
        self._data_flux = fit.like.data_flux
        if fit.like.has_data_covmatrix:
            self._has_covmatrix = True
            self._covmatrix = fit.like.data_covmatrix
            self._invcovmatrix = fit.like.data_invcovmatrix
            self._data_flux_unc = fit.like.data_flux_unc  # diag of cov
        else:
            self._has_covmatrix = False
            self._data_flux_unc = fit.like.data_flux_unc
            if hasattr(self, '_covmatrix'):
                del self._covmatrix
            if hasattr(self, '_invcovmatrix'):
                del self._invcovmatrix

        # Chain variables
        self.chain = fit.sampler.chain
        self.lnprobability = fit.sampler.lnprobability

        # Set up central values for all parameters
        self.par_central_values = np.array([self.par_cen(i) for i in range(5)])

        #Get the best fit point
        idxmax_flat = self.lnprobability.argmax()
        idxmax = np.unravel_index(idxmax_flat, self.lnprobability.shape)
        self._best_fit = (self.chain[idxmax[0], idxmax[1], :],
                          self.lnprobability[idxmax[0], idxmax[1]],
                          idxmax)

        #We don't have any ancillary variables at this point
        self._has_lir = False
        self._lir_min = None
        self._lir_max = None
        if hasattr(self, 'lir'):
            del self.lir
        self._has_dustmass = False
        self._kappa = None
        self._kappa_wave = None
        if hasattr(self, 'dustmass'):
            del self.dustmass
        self._has_peaklambda = False
        if hasattr(self, 'peaklambda'):
            del self.peaklambda

    @property
    def redshift(self):
        """ Redshift"""
        return self._z

    @property
    def opthin(self):
        """ Was an optically thin model assumed?"""
        if not self._fitset:
            return None
        return self._opthin

    @property
    def noalpha(self):
        """ Was alpha not used?"""
        if not self._fitset:
            return None
        return self._noalpha

    @property
    def wavenorm(self):
        """ Wavelength (obs frame) of normalization"""
        if not self._fitset:
            return None
        return self._wavenorm

    @property
    def response_integrate(self):
        """Was response integration in use?"""
        if not self._fitset:
            return None
        return self._response_integrate

    @property
    def cosmo_type(self):
        """ Type of cosmology used"""
        return self._cosmo_type

    @property
    def cosmology(self):
        """ Return the current cosmology"""
        if not self._cosmo_type in astropy.cosmology.parameters.available:
            raise ValueError("Unknown cosmology type: {:s}".
                             format(self._cosmo_type))
        return getattr(astropy.cosmology, self._cosmo_type)

    @property
    def lumdist(self):
        """ Returns luminosity distance to object"""
        if self._has_lumdist:
            return self._lumdist
        else:
            if self._z is None:
                raise Exception("Need redshift to be set to compute lumdist")
            if self._z <= -1:  # Not okay!
                errmsg = "Redshift is less than -1: {:f}"
                raise ValueError(errmsg.format(self._z))
            dl = self.cosmology.luminosity_distance(self._z)
            if isinstance(dl, u.Quantity):
                return dl
            else:
                return u.Quantity(dl, u.Mpc)

    @property
    def best_fit(self):
        """ Gets the best fitting point that occurred during the fit

        Returns
        -------
        tup : tuple
          A tuple of the parameters, the log probability, and the
          index into lnprobability
        """

        if not self._fitset:
            return None
        return self._best_fit

    @property
    def best_fit_chisq(self):
        """ Get the chisq of the best fitting point.

        Returns
        -------
        chisq : float
          The chi2 of the best fitting point.
        """

        if not self._fitset:
            return None
        return -2.0 * self._best_fit[1]

    def best_fit_sed(self, wave):
        """ Get the best fitting SED

        Parameters
        ----------
        wave : ndarray
          Wavelengths the sed is desired at, in microns

        Returns
        -------
        sed : ndarray
          The sed corresponding to the best fitting parameters at
          the wavelengths specified by wave
         """

        if not self._fitset:
            return None
        pars = self._best_fit[0]
        sed = modified_blackbody(pars[0], pars[1], pars[2], pars[3],
                                 pars[4], wavenorm=self._wavenorm,
                                 noalpha=self._noalpha,
                                 opthin=self._opthin)
        return sed(wave)

    @property
    def data(self):
        """ Get tuple of data wavelengths, flux densities, uncertainties"""

        if not self._fitset:
            return None
        return (self._data_wave, self._data_flux, self._data_flux_unc)

    @property
    def covmatrix(self):
        """ Get covariance matrix, or None if none present"""

        if not self._fitset or not self._has_covmatrix:
            return None
        return self._covmatrix

    def _parcen_internal(self, array, percentile, lowlim=None,
                         uplim=None):
        """ Internal computation of parameter central limits

        Parameters
        ----------
        array : ndarray
          Input array

        percentile : float
          Percentile for computation (0-100)

        lowlim : float
          Smallest value to allow in computation

        uplim : float
          Largest value to allow in computation

        Returns
        -------
        res : ndarray
          Mean, upper uncertainty, lower uncertainty.
        """

        if not self._fitset:
            raise Exception("Fit not available")

        pcnt = float(percentile)
        if pcnt < 0 or pcnt > 100:
            raise ValueError("Invalid percentile {:f}".format(pcnt))

        aint = copy.deepcopy(array)

        # Slice off ends if needed
        if (not lowlim is None) or (not uplim is None):
            if lowlim is None:
                cond = (aint <= float(uplim)).nonzero()[0]
            elif uplim is None:
                cond = (aint >= float(lowlim)).nonzero()[0]
            else:
                cond = np.logical_and(aint >= float(lowlim),
                                      aint <= float(uplim)).nonzero()[0]
            if len(cond) == 0:
                raise Exception("No elements survive lower/upper "
                                "limit clipping")
            if len(cond) != len(aint):
                aint = aint[cond]

        aint.sort()
        mnval = np.mean(aint)
        pval = (1.0 - 0.01 * pcnt) / 2
        na = len(aint)
        lowidx = int(round(pval * na))
        lowval = aint[lowidx]
        upidx = int(round((1.0 - pval) * na))
        upval = aint[upidx]
        return np.array([mnval, upval-mnval, mnval-lowval])

    def parameter_chain(self, param):
        """ Gets flattened chain for parameter

        Parameters
        ----------
        param : int or string
          Parameter specification

        Returns
        -------
        chain : ndarray
          Flattened chain for specified parameter
        """

        if not self._fitset:
            return None

        if isinstance(param, str):
            paridx = self._param_order[param.lower()]
        else:
            paridx = int(param)
            if paridx < 0 or paridx > 5:
                raise ValueError("invalid parameter index {:d}".format(paridx))

        return self.chain[:, :, paridx].flatten()

    def par_cen(self, param, percentile=68.3, lowlim=None, uplim=None):
        """ Gets the central confidence interval for the parameter

        Parameters
        ----------
        param : int or string
          Parameter specification.  Either an index into
          the parameter list, or a string name for the
          parameter.

        percentile : float
          Percentile of uncertainties to use.  1 sigma is 68.3,
          2 sigma is 95.4, etc.

        lowlim : float
          Lower limit for parameter to include in computation

        uplim : float
          Lower limit for parameter to include in computation

        Returns
        -------
        ret : ndarray
          A 3 element ndarray of the mean value, upper confidence limit,
          and lower confidence limit.
          Percentile of limit to compute
        """

        if not self._fitset:
            return None
        if percentile <= 0 or percentile >= 100.0:
            raise ValueError("percentile needs to be between 0 and 100")

        return self._parcen_internal(self.parameter_chain(param),
                                     percentile, lowlim=lowlim, uplim=uplim)

    def par_lowlim(self, param, percentile=68.3):
        """ Gets the lower limit for the parameter

        param : int or string
          Parameter specification.  Either an index into
          the parameter list, or a string name for the
          parameter.

        percentile : float
          Confidence level associated with limit.

        Returns
        -------
        lowlim : float
          Lower limit on parameter
        """

        if not self._fitset:
            return None
        if isinstance(param, str):
            paridx = self._param_order[param.lower()]
        else:
            paridx = int(param)
            if paridx < 0 or paridx > 5:
                raise ValueError("invalid parameter index {:d}".format(paridx))

        if percentile <= 0 or percentile >= 100.0:
            raise ValueError("percentile needs to be between 0 and 100")

        svals = self.parameter_chain(paridx)
        svals.sort()
        return svals[round((1.0 - 0.01 * percentile) * len(svals))]

    def par_uplim(self, param, percentile=68.3):
        """ Gets the upper limit for the parameter

        param : int or string
          Parameter specification.  Either an index into
          the parameter list, or a string name for the
          parameter.

        percentile : float
          Confidence level associated with limit.

        Returns
        -------
        uplim : float
          Upper limit on parameter
        """

        if not self._fitset:
            return None
        if isinstance(param, str):
            paridx = self._param_order[param.lower()]
        else:
            paridx = int(param)
            if paridx < 0 or paridx > 5:
                raise ValueError("invalid parameter index {:d}".format(paridx))

        if percentile <= 0 or percentile >= 100.0:
            raise ValueError("percentile needs to be between 0 and 100")

        svals = self.chain[:, :, paridx].flatten()
        svals.sort()
        return svals[round(0.01 * percentile * len(svals))]

    @property
    def has_peaklambda(self):
        return self._has_peaklambda

    @property
    def peaklambda_chain(self):
        """ Get flattened chain of peak lambda values in microns"""

        if not self._has_peaklambda:
            return None
        return self.peaklambda.flatten()

    def peaklambda_cen(self, percentile=68.3, lowlim=None, uplim=None):
        """ Gets the central confidence interval for the peak lambda.

        Parameters
        ----------
        percentile : float
          The percentile to use when computing the uncertainties.

        lowlim : float
          Smallest value to allow in computation

        uplim : float
          Largest value to allow in computation

        Returns
        -------
        res: ndarray
          A 3 element array of the central value, upper uncertainty,
          and lower uncertainty of the peak observer frame
          wavelength in microns.
        """

        if not self._has_peaklambda:
            return None
        return self._parcen_internal(self.peaklambda.flatten(), percentile,
                                     lowlim=lowlim, uplim=uplim)

    def _map_chain(self, func, maxidx=None, **kwargs):
        """ Computes func over all values in the chains.

        func : function
          Function to apply

        maxidx : int
          Maximum index in each walker to use.  If None, use all

        This avoid recomputing the value when a step is equal to its
        previous value (unlike, say, map).
        """

        shp = self.chain.shape[0:2]
        steps = shp[1]
        if not maxidx is None and maxidx < shp[1]:
            shp[1] = maxidx

        retvar = np.empty(shp, dtype=np.float)
        for walkidx in range(shp[0]):
            # Do first step
            prevstep = self.chain[walkidx, 0, :]
            retvar[walkidx, 0] = func(prevstep, **kwargs)

            #Now other steps
            for stepidx in range(1, shp[1]):
                currstep = self.chain[walkidx, stepidx, :]
                if np.allclose(prevstep, currstep):
                    # Repeat, so avoid re-computation
                    retvar[walkidx, stepidx] = retvar[walkidx, stepidx - 1]
                else:
                    retvar[walkidx, stepidx] = func(currstep, **kwargs)
                    prevstep = currstep

        return retvar

    def compute_peaklambda(self):
        """ Compute the observer frame wavelength of peak emission
        in microns from chain"""

        def peaklambda_inner(step, opthin=False, noalpha=False):
            sed = modified_blackbody(step[0], step[1], step[2], step[3],
                                     step[4], opthin=opthin,
                                     noalpha=noalpha)
            return sed.max_wave()

        self.peaklambda = self._map_chain(peaklambda_inner)
        self._has_peaklambda = True

    @property
    def has_lir(self):
        return self._has_lir

    @property
    def lir_wavelength(self):
        """ Range, in microns, of L_IR integration"""
        return (self._lir_min, self._lir_max)

    @property
    def lir_chain(self):
        """ Get flattened chain of l_ir values in 10^12 solar luminosities."""

        if not self._has_lir:
            return None
        return self.lir.flatten()

    def lir_cen(self, percentile=68.3, lowlim=None, uplim=None):
        """ Gets the central confidence interval for L_IR

        Parameters
        ----------
        percentile : float
          The percentile to use when computing the uncertainties.

        lowlim : float
          Smallest value to allow in computation

        uplim : float
          Largest value to allow in computation

        Returns
        -------
        res : ndarray
          A 3 element array of the central value, upper uncertainty,
          and lower uncertainty of the IR luminosity (8-1000um)
          in 10^12 solar luminosities, or None if the L_IR has
          not been computed
        """
        if not self._has_lir:
            return None
        return self._parcen_internal(self.lir.flatten(), percentile,
                                     lowlim=lowlim, uplim=uplim)

    def compute_lir(self, wavemin=8.0, wavemax=1000.0, maxidx=None):
        """ Computes LIR from chain in 10^12 L_sun.

        Parameters
        ----------
        wavemin : float
          Minimum wavelength of L_IR integration, in microns

        wavemax : float
          Maximum wavelength of L_IR integration, in microns

        maxidx : int
          Maximum index in each walker to use.

        Notes
        -----
        The traditional definition of L_IR is 8-1000um.  Some authors use
        42.5-112.5um instead.
        """

        if not self._fitset:
            raise Exception("Fit results not loaded")
        if self._z is None:
            raise Exception("Redshift must be set to compute L_IR")

        self._lir_min = float(wavemin)
        self._lir_max = float(wavemax)
        if self._lir_min <= 0:
            raise ValueError("Invalid wavemin: {:f}".format(self._lir_min))
        if self._lir_max <= 0:
            raise ValueError("Invalid wavemax: {:f}".format(self._lir_max))
        # Get them in ascending order
        if self._lir_min > self._lir_max:
            self._lir_min, self._lir_max = self._lir_max, self._lir_min

        # 4*pi*dl^2/L_sun in cgs -- so the output will be in
        # solar luminosities; the prefactor is
        # 4 * pi * mpc_to_cm^2/L_sun
        dl = self.lumdist.value  # in Mpc
        lirprefac = 3.11749657e4 * dl**2  # Also converts to 10^12 lsolar

        # L_IR defined as between 8 and 1000 microns (rest)
        integrator = mbb_freqint(self._z, self._lir_min, self._lir_max,
                                 opthin=self.opthin,
                                 noalpha=self.noalpha)

        self.lir = lirprefac * self._map_chain(integrator, maxidx=maxidx)
        self._has_lir = True

    @property
    def has_dustmass(self):
        """ Has dustmass been computed?"""
        return self._has_dustmass

    @property
    def dust_kappa(self):
        """ Dust opacity in m^2 kg^-1"""
        return self._kappa

    @property
    def dust_kappa_wavelength(self):
        """ Rest wavelength dust opacity is defined at, in microns"""
        return self._kappa_wave

    @property
    def dustmass_chain(self):
        """ Get flattened chain of dustmass values in 10^8 solar masses"""

        if not self._has_dustmass:
            return None
        return self.dustmass.flatten()

    def dustmass_cen(self, percentile=68.3, lowlim=None, uplim=None):
        """ Gets the central confidence interval for dustmass.

        Parameters
        ----------
        percentile: float
          The percentile to use when computing the uncertainties.

        lowlim: float
          Smallest value to allow in computation

        uplim: float
          Largest value to allow in computation

        Returns
        -------
        res: ndarray
          A 3 element array of the central value, upper uncertainty,
          and lower uncertainty of the dust mass in 10^8 solar masses,
          or None if not computed.
        """

        if not self._has_dustmass:
            return None
        return self._parcen_internal(self.dustmass.flatten(), percentile,
                                     lowlim=lowlim, uplim=uplim)

    def _dmass_calc(self, step, opz, bnu_fac, temp_fac, knu_fac,
                    opthin, dl2):
        """Internal function to comput dustmass in 10^8 M_sun,
        given various pre-computed values"""

        msolar8 = 1.97792e41  # mass of the sun*10^8 in g
        T = step[0] * opz
        beta = step[1]
        S_nu = step[4] * 1e-26  # to erg / s-cm^2-Hz from mJy
        B_nu = bnu_fac / math.expm1(temp_fac / T)  # Planck function
        # Scale kappa with freq (obs frame ok).  Factor of 10 is
        #  m^2 kg^-1 -> cm^2 g^-1 conversion
        K_nu = 10.0 * self._kappa * knu_fac**(-beta)
        dustmass = dl2 * S_nu / (opz * K_nu * B_nu * msolar8)
        if not opthin:
            tau_nu = (step[2] / self._wavenorm)**beta
            op_fac = - tau_nu / math.expm1(-tau_nu)
            dustmass *= op_fac
        return dustmass

    def compute_dustmass(self, kappa=2.64, kappa_wave=125.0, maxidx=None):
        """Compute dust mass in 10^8 M_sun from chain

        Parameters
        ----------
        kappa : float
          Dust opacity coeffient in m^2 kg^-1.  The default value,
          2.64, is from Dunne et al. 2003 at 125um.

        kappa_wave : float
          The rest frame wavelength that kappa is defined at, in microns.
          The default value corresponds to the kappa default.

        maxidx : int
          Maximum index in each walker to use.  Ignored if threading.
        """

        if not self._fitset:
            raise Exception("Fit not processed")
        if self._z is None:
            raise Exception("Redshift must be set to compute dust mass")

        self._kappa = float(kappa)
        self._kappa_wave = float(kappa_wave)

        if self._kappa <= 0:
            errmsg = "Invalid (non-positive) kappa {:f}"
            raise ValueError(errmsg.format(self._kappa))
        if self._kappa_wave <= 0:
            errmsg = "Invalid (non-positive) kappa wavelength {:f}"
            raise ValueError(errmsg.format(self._kappa_wave))

        dl = self.lumdist.to(u.cm).value
        dl2 = dl**2
        opz = 1.0 + self._z

        wavenorm_rest = self.wavenorm / opz  # in um
        nunorm_rest = 299792458e6 / wavenorm_rest  # in Hz

        # Precompute some quantities for evaluating the Planck function
        # h nu / k and 2 h nu^3 / c^2
        temp_fac = 6.6260693e-27 * nunorm_rest / 1.38065e-16  # h nu / k
        bnu_fac = 2 * 6.6260693e-27 * nunorm_rest**3 / 299792458e2**2

        # Work out dust opacity factor.
        knu_fac = wavenorm_rest / self._kappa_wave

        msolar8 = 1.97792e41  # mass of the sun*10^8 in g

        # Use closure
        def dmass_inner(step):
            return self._dmass_calc(step, opz, bnu_fac, temp_fac, knu_fac,
                                    self.opthin, dl2)

        self.dustmass = self._map_chain(dmass_inner, maxidx=maxidx)
        self._has_dustmass = True

    def choice(self, nsamples=1, getpeaklambda=False, getlir=False,
               getdustmass=False):
        """ Get a random sample from the parameter chain, returning
        the parameters.

        Parameters
        ----------
        nsamples : int
          Number of samples to generate.

        getpeaklambda : bool
          If true, adds peaklambda sample, returning a tuple

        getlir : bool
          If true, adds lir sample, returning a tuple.  Peaklambda
          is added first

        getdustmass : bool
          If true, adds a dustmass sample.  Peaklambda and lir are added
          first.

        Returns
        -------
        sample : tuple
          A tuple with the set of parameters as a 5 by nsamples array.
          If getpeaklambda, getlir, or getdustmass are set, additional
          ndarrays of length nsamples are appended, in that order.
          These are flattened into a 5 element array and floats if
          nsamples is 1.
        """

        if not self._fitset:
            return None
        if nsamples == 0:
            return None

        #Make sure things are calculated
        if getpeaklambda and not self._has_peaklambda:
            raise Exception("Peak lambda not computed")
        if getlir and not self._has_lir:
            raise Exception("LIR not computed")
        if getdustmass and not self._has_dustmass:
            raise Exception("Dustmass not computed")

        if nsamples == 1:
            # It's ugly, but still cleaner to do this in two branches
            #  depending on nsamples

            # Generate index into chain
            idx_walker = np.random.randint(0, self.chain.shape[0])
            idx_step = np.random.randint(0, self.chain.shape[1])

            # Get samples at that index
            rettup = (self.chain[idx_walker, idx_step, :],)
            if getpeaklambda:
                rettup += (self.peaklambda[idx_walker, idx_step],)
            if getlir:
                rettup += (self.lir[idx_walker, idx_step],)
            if getdustmass:
                rettup += (self.dustmass[idx_walker, idx_step],)
        else:
            #Multi-sample case
            rettup = (np.empty((nsamples, 5), dtype=np.float64),)
            runidx = 1
            if getpeaklambda:
                peakidx = runidx
                rettup += (np.empty(nsamples, dtype=np.float64),)
                runidx += 1
            if getlir:
                liridx = runidx
                rettup += (np.empty(nsamples, dtype=np.float64),)
                runidx += 1
            if getdustmass:
                dustidx = runidx
                rettup += (np.empty(nsamples, dtype=np.float64),)

            # Generate index into chain
            for idx in range(nsamples):
                idx_walker = np.random.randint(0, self.chain.shape[0])
                idx_step = np.random.randint(0, self.chain.shape[1])

                rettup[0][idx, :] = self.chain[idx_walker, idx_step, :]
                if getpeaklambda:
                    rettup[peakidx][idx] = self.peaklambda[idx_walker,
                                                           idx_step]
                if getlir:
                    rettup[liridx][idx] = self.lir[idx_walker, idx_step]
                if getdustmass:
                    rettup[dustidx][idx] = self.dustmass[idx_walker, idx_step]

        return rettup

    def _predict_flux(self, spec, maxidx=None):
        """ Predict flux density at a given wavelength from the fit

        Parameters
        ----------
        spec : float or string
          Either wavelength to predict flux at or name of an instrument
          response function to integrate.

        maxidx : int
          Maximum index in each walker to use.

        Returns
        -------
        fluxchain : ndarray
          Array of predictions.  If spec is a float, this is
          the sed flux at that value.  If it is a string, it is
          the response predicted for that response function name.
          So, for example, if spec = 'SPIRE_250um' it will be
          the predicted flux integrated through the Herschel-SPIRE
          250um filter function -- if that was available to the fit.
        """

        if isinstance(spec, basestring):
            if not self._response_integrate:
                raise Exception("Asked for response integration, but no "
                                "response functions available from original "
                                "fit")
            if not spec in self._responsewheel:
                errmsg = "Do not have response function matching {:s}"
                raise ValueError(errmsg.format(spec))
            resp = self._responsewheel[spec]
            doing_response = True
        else:
            wv = float(spec)
            if (wv <= 0):
                raise ValueError("Invalid wavelength {:f}".format(wv))
            doing_response = False

        def predflux_inner(step):
            sed = modified_blackbody(step[0], step[1], step[2],
                                     step[3], step[4],
                                     opthin=self.opthin,
                                     noalpha=self.noalpha)
            if doing_response:
                return resp(sed)
            else:
                return sed(wv)

        return self._map_chain(predflux_inner, maxidx=maxidx)

    def predflux_cen(self, spec, percentile=68.3, maxidx=None,
                     lowlim=None, uplim=None):
        """ Gets the central confidence interval for predicted flux

        Parameters
        ----------
        spec : float or string
          Either wavelength to predict flux at or name of an instrument
          response function to integrate.

        percentile : float
          The percentile to use when computing the uncertainties.

        maxidx : int
          Maximum index in each walker to use.

        lowlim : float
          Smallest value to allow in computation

        uplim : float
          Largest value to allow in computation

        Returns
        -------
        res : ndarray
          A 3 element array of the central value, upper uncertainty,
          and lower uncertainty of the predicted flux in mJy.
          If spec was a float, this is the sed flux at that value.
          If it was a string, it is the response predicted for that
          response function name. So, for example, if spec = 'SPIRE_250um'
          it will be the predicted flux integrated through the Herschel-SPIRE
          250um filter function -- if that was available to the fit.
        """

        if not self._fitset:
            return None
        pflux = self._predict_flux(spec, maxidx)

        return self._parcen_internal(pflux.flatten(), percentile,
                                     lowlim=lowlim, uplim=uplim)

    def writeToHDF5(self, filename):
        """ Serialize the results of the fit to an HDF5 file"""

        import h5py

        if not self._fitset:
            raise Exception("Fit not processed")

        f = h5py.File(filename, 'w')  # Overwrite if pre-existing

        # Write fit specification information as top level attributes
        if not self._z is None:
            f.attrs["z"] = self._z
        f.attrs["Noalpha"] = self._noalpha
        f.attrs["Opthin"] = self._opthin
        f.attrs["Nwalkers"] = self._nwalkers
        f.attrs["Wavenorm"] = self._wavenorm
        f.attrs["Lowlim"] = self._lowlim
        f.attrs["HasUplim"] = self._has_uplim
        f.attrs["Uplim"] = self._uplim
        f.attrs["HasGaussianPrior"] = self._has_gprior
        f.attrs["GaussianPriorMean"] = self._gprior_mean
        f.attrs["GaussianPriorSigma"] = self._gprior_sigma
        f.attrs["GaussianPriorIVar"] = self._gprior_ivar
        f.attrs["ResponseIntegrate"] = self._response_integrate
        f.attrs["Fixed"] = self._fixed
        f.attrs["Ndata"] = self._ndata

        # Responses if present
        if self._response_integrate:
            gr = f.create_group("Responses")
            self._responsewheel.writeToHDF5(gr)

        # Data group
        gd = f.create_group("Data")
        gd.attrs["WaveUnits"] = "microns"
        gd.attrs["FluxUnits"] = "mJy"
        gd.create_dataset("Wave", data=self._data_wave)
        gd.create_dataset("FluxDensity", data=self._data_flux)
        gd.create_dataset("FluxDensityUnc", data=self._data_flux_unc)
        if self._has_covmatrix:
            gd.create_dataset("Covmatrix", data=self._covmatrix)
            gd.create_dataset("InvCovmatrix", data=self._invcovmatrix)

        # Chain results
        gc = f.create_group("Chain")
        gc.create_dataset("ParamCentralValues", data=self.par_central_values)
        gc.create_dataset("Chain", data=self.chain)
        gc.create_dataset("LogLike", data=self.lnprobability)
        gc.create_dataset("BestFitParams", data=self._best_fit[0])
        gc.create_dataset("BestFitLogLike", data=np.array(self._best_fit[1]))
        gc.create_dataset("BestFitIndex", data=np.array(self._best_fit[2]))

        # Ancillary variables (L_IR, etc.)
        ga = f.create_group("Ancillary")
        ga.attrs["cosmo_type"] = self._cosmo_type
        if self._has_lumdist:
            ga.attrs["lumdist"] = self._lumdist.value  # in Mpc
        if self._has_lir:
            ga.attrs["LirMin"] = self._lir_min
            ga.attrs["LirMax"] = self._lir_max
            ga.create_dataset("Lir", data=self.lir)
        if self._has_dustmass:
            ga.attrs["Kappa"] = self._kappa
            ga.attrs["KappaWave"] = self._kappa_wave
            ga.create_dataset("Dustmass", data=self.dustmass)
        if self._has_peaklambda:
            ga.create_dataset("PeakLambda", data=self.peaklambda)

        f.close()

    def readFromHDF5(self, filename):
        """ Restores from an HDF 5 file"""

        import h5py
        f = h5py.File(filename, 'r')

        if "z" in f.attrs:
            self._z = f.attrs["z"]
        else:
            self._z = None

        self._noalpha = f.attrs["Noalpha"]
        self._opthin = f.attrs["Opthin"]
        self._nwalkers = f.attrs["Nwalkers"]
        self._wavenorm = f.attrs["Wavenorm"]
        self._lowlim = f.attrs["Lowlim"]
        self._has_uplim = f.attrs["HasUplim"]
        self._uplim = f.attrs["Uplim"]
        self._has_gprior = f.attrs["HasGaussianPrior"]
        self._gprior_mean = f.attrs["GaussianPriorMean"]
        self._gprior_sigma = f.attrs["GaussianPriorSigma"]
        self._gprior_ivar = f.attrs["GaussianPriorIVar"]
        self._response_integrate = f.attrs["ResponseIntegrate"]
        self._fixed = f.attrs["Fixed"]
        self._ndata = f.attrs["Ndata"]

        if self._response_integrate:
            if not "Responses" in f:
                errmsg = "Didn't find expected responses in {:s}"
                raise ValueError(errmsg.format(filename))
            self._responsewheel = response_set()
            self._responsewheel.readFromHDF5(f["Responses"])

        gd = f["Data"]
        self._data_wave = gd["Wave"][...]
        self._data_flux = gd["FluxDensity"][...]
        self._data_flux_unc = gd["FluxDensityUnc"][...]
        if "Covmatrix" in gd:
            self._has_covmatrix = True
            self._covmatrix = gd["Covmatrix"][...]
            self._invcovmatrix = gd["InvCovmatrix"][...]
        else:
            self._has_covmatrix = False
            if hasattr(self, "_covmatrix"):
                del self._covmatrix
            if hasattr(self, "_invcovmatrix"):
                del self._invcovmatrix

        gc = f["Chain"]
        self.par_central_values = gc["ParamCentralValues"][...]
        self.chain = gc["Chain"][...]
        self.lnprobability = gc["LogLike"][...]
        self._best_fit = (gc["BestFitParams"][...],
                          gc["BestFitLogLike"][()],
                          gc["BestFitIndex"][()])

        ga = f["Ancillary"]

        if "cosmo_type" in ga.attrs:
            self._cosmo_type = ga.attrs["cosmo_type"]
        if "lumdist" in ga.attrs:
            self._has_lumdist = True
            self._lumdist = u.Quantity(ga.attrs["lumdist"], u.Mpc)
        else:
            self._has_lumdist = False
        if "Lir" in ga.attrs:
            self._lir_min = ga.attrs["LirMin"]
            self._lir_max = ga.attrs["LirMax"]
            self.lir = ga["Lir"][...]
            self._has_lir = True
        else:
            self._has_lir = False
            self._lir_min = None
            self._lir_max = None
            if hasattr(self, 'lir'):
                del self.lir

        if "Dustmass" in ga:
            self._kappa = ga.attrs["Kappa"]
            self._kappa_wave = ga.attrs["KappaWave"]
            self.dustmass = ga["Dustmass"][...]
            self._has_dustmass = True
        else:
            self._has_dustmass = False
            self._kappa = None
            self._kappa_wave = None
            if hasattr(self, 'dustmass'):
                del self.dustmass

        if "PeakLambda" in ga:
            self._has_peaklambda = True
            self.peaklambda = ga["PeakLambda"][...]
        else:
            self._has_peaklambda = False
            if hasattr(self, 'peaklambda'):
                del self.peaklambda

        self._fitset = True

        f.close()

    def __str__(self):
        """ String representation of results"""

        if not self._fitset:
            return "<Uninitialized mbb_results object>"

        idx = [0, 1, 4]
        tag = ["T/(1+z)", "beta", "fnorm"]
        units = ["[K]", "", "[mJy]"]
        retstr = ""

        for i, tg, unit in zip(idx, tag, units):
            retstr += "{:s}: ".format(tg)
            if self._fixed[i]:
                val = self.chain[:, :, i].mean()
                retstr += "{:0.2f} (fixed)\n".format(val)
            else:
                msg = "{:0.2f} +{:0.2f} -{:0.2f}"
                retstr += msg.format(self.par_central_values[i][0],
                                     self.par_central_values[i][1],
                                     self.par_central_values[i][2])
                retstr += " (low lim: {:0.2f}".format(self._lowlim[i])
                if self._has_uplim[i]:
                    retstr += " upper lim: {:0.2f}".format(self._uplim[i])
                if self._has_gprior[i]:
                    msg = " prior: {:0.2f} {:0.2f}"
                    retstr += msg.format(self._gprior_mean[i],
                                         self._gprior_sigma[i])
                retstr += ") {:s}\n".format(unit)

        if not self._opthin:
            if self._fixed[2]:
                val = self.chain[:, :, 2].mean()
                retstr += "lambda0 (1+z): {:0.2f} (fixed) [um]\n".format(val)
            else:
                msg = "lambda0 (1+z): {:0.2f} +{:0.2f} -{:0.2f}"
                retstr += msg.format(self.par_central_values[2][0],
                                     self.par_central_values[2][1],
                                     self.par_central_values[2][2])
                retstr += " (low lim: {:0.2f}".format(self._lowlim[2])
                if self._has_uplim[2]:
                    retstr += " upper lim: {:0.2f}".format(self._uplim[2])
                if self._has_gprior[2]:
                    msg = " prior: {:0.2f} {:0.2f}"
                    retstr += msg.format(self._gprior_mean[2],
                                         self._gprior_sigma[2])
                retstr += ") [um]\n"
        else:
            retstr += "Optically thin case assumed\n"

        if not self._noalpha:
            if self._fixed[3]:
                val = self.chain[:, :, 3].mean()
                retstr += "alpha: {:0.2f} (fixed)\n".format(val)
            else:
                msg = "alpha: {:0.2f} +{:0.2f} -{:0.2f}"
                retstr += msg.format(self.par_central_values[3][0],
                                     self.par_central_values[3][1],
                                     self.par_central_values[3][2])
                retstr += " (low lim: {:0.2f}".format(self._lowlim[3])
                if self._has_uplim[3]:
                    retstr += " upper lim: {:0.2f}".format(self._uplim[3])
                if self._has_gprior[3]:
                    msg = " prior: {:0.2f} {:0.2f}"
                    retstr += msg.format(self._gprior_mean[3],
                                         self._gprior_sigma[3])
                retstr += ")\n"
        else:
            retstr += "Alpha not used\n"

        # Lambda peak prior
        if self._has_uplim[5] or self._has_gprior[5]:
            retstr += "Lambda_peak prior"
            if self._has_uplim[5]:
                retstr += " upper lim: {:0.2f}".format(self._uplim[5])
            if self._has_gprior[5]:
                msg = " prior: {:0.2f} {:0.2f}"
                retstr += msg.format(self._gprior_mean[5],
                                     self._gprior_sigma[5])
            retstr += "\n"

        if self.has_peaklambda:
            pstr = "Lambda peak: {:0.1f} +{:0.1f} -{:0.1f} [um]\n"
            retstr += pstr.format(*self.peaklambda_cen())

        if self.has_lir:
            args = self.lir_wavelength + tuple(self.lir_cen())
            lirstr = "L_IR({:0.1f} to {:0.1f}um): {:0.2f} "\
                     "+{:0.2f} -{:0.2f} [10^12 L_sun]\n"
            retstr += lirstr.format(*args)

        if self.has_dustmass:
            args = (self.dust_kappa, self.dust_kappa_wavelength) + \
                tuple(self.dustmass_cen())
            duststr = "M_d(kappa={0:0.2f}, lam={1:0.1f}um): {2:0.2f} "\
                      "+{3:0.2f} -{4:0.2f} [10^8 M_sun]\n"
            retstr += duststr.format(*args)

        retstr += "Number of data points: {:d}\n".format(self._ndata)
        wstr = "ChiSquare of best fit point: {:0.2f}"
        retstr += wstr.format(self.best_fit_chisq)

        return retstr

############################################################


class mbb_freqint(object):
    """ Does frequency integration"""
    # The idea is to allow this to be multiprocessed

    def __init__(self, redshift, lammin, lammax, opthin=False,
                 noalpha=False):
        """
        Parameters
        __________
        redshift : float
           Redshift of the object

        lammin : float
           Minimum wavelength of frequency integral, in microns

        lammax : float
           Maximum wavelength of frequency integral, in microns

        opthin : bool
           Is the integration optically thin?

        noalpha : bool
           Ignore alpha
        """

        self._redshift = float(redshift)
        self._lammin = float(lammin)
        self._lammax = float(lammax)
        self._opthin = bool(opthin)
        self._noalpha = bool(noalpha)

        if self._redshift < 0:
            errmsg = "Invalid (negative) redshift: {:f}"
            raise Exception(errmsg.format(self._redshift))
        if self._lammin <= 0:
            errmsg = "Invalid (non-positive) lammin: {:f}"
            raise Exception(errmsg.format(self._lammin))
        if self._lammax <= 0:
            errmsg = "Invalid (non-positive) lammax: {:f}"
            raise Exception(errmsg.format(self._lammax))
        if self._lammin > self._lammax:
            self._lammin, self._lammax = self._lammax, self._lammin

        opz = 1.0 + self._redshift
        self._minwave_obs = self._lammin * opz
        self._maxwave_obs = self._lammax * opz

    def __call__(self, params):
        """ Evaluates frequency integral.

        Parameters
        ----------
        params : ndarray
          Array of parameter values in order T, beta, lambda0, alpha, fnorm
        """

        mbb = modified_blackbody(params[0], params[1], params[2],
                                 params[3], params[4], opthin=self._opthin,
                                 noalpha=self._noalpha)
        return mbb.freq_integrate(self._minwave_obs, self._maxwave_obs)
