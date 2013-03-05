from __future__ import print_function

import numpy as np
import emcee
import math
import multiprocessing
from .modified_blackbody import modified_blackbody
from .likelihood import likelihood
import copy

__all__ = ["mbb_fit", "mbb_fit_results"]

#hack for basestring
try:
    basestring
except:
    #Python 3
    basestring = str

############################################################

# This class holds the results.  Why not just save the
# fit class?  Because mbb_fit (the fitting class) can involve a 
# multiprocessing pool, which can't be pickled.  So instead 
# package the results up in this, which also adds methods for 
# finding central limits, the best fit point, etc., etc.
# This class also allows the computation of ancillary quantities
# like L_IR, etc. rather than doing them in the main fit.

class mbb_fit_results(object):
    """Holds results of fit"""

    """ Parameter order dictionary.  Lowercased."""
    _param_order = {'t': 0, 't/(1+z)': 0, 'beta': 1, 'lambda0': 2,
                    'lambda0*(1+z)': 2, 'lambda_0': 2, 'lambda_0*(1+z)': 2,
                    'alpha': 3, 'fnorm': 4, 'f500': 4}

    def __init__(self, fit, redshift=None):
        """
        Parameters
        ----------
        fit : mbb_fit
          Fit object
          
        redshift : float
          Redshift of source.  Necessary if you plan to compute
          dustmass, L_IR, or L_AGN
        """

        import copy

        if not isinstance(fit, mbb_fit):
            raise ValueError("Input is not of type mbb_fit")

        if redshift is None:
            self._z = None
        else:
            self._z = float(redshift)

        self.like = copy.deepcopy(fit.like)
        self.chain = copy.copy(fit.sampler.chain)
        self.lnprobability = copy.copy(fit.sampler.lnprobability)
        self.fixed = copy.copy(fit._fixed)

        self.par_central_values = [self.par_cen(i) for i in range(5)]

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
        self._has_dustmass = False
        self._kappa = None
        self._kappa_wave = None
        self._has_peaklambda = False

    @property
    def redshift(self):
        return self._z

    @property
    def opthin(self):
        return self.like.opthin

    @property
    def noalpha(self):
        return self.like.noalpha

    @property
    def wavenorm(self):
        return self.like.wavenorm

    @property
    def response_integrate(self):
        """Was response integration in use?"""
        return self.like.response_integrate

    @property
    def best_fit(self):
        """ Gets the best fitting point that occurred during the fit

        Returns
        -------
        tup : tuple
          A tuple of the parameters, the log probability, and the
          index into lnprobability
        """

        return self._best_fit

    @property
    def best_fit_chisq(self):
        """ Get the chisq of the best fitting point.

        Returns
        -------
        chisq : float
          The chi2 of the best fitting point.
        """
        
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
        
        return self.like.get_sed(self._best_fit[0], wave)

    @property
    def data(self):
        """ Get tuple of data wavelengths, flux densities, uncertainties"""
        if not self.like.data_read: return None
        return (self.like.data_wave, self.like.data_flux,
                self.like.data_flux_unc)

    @property
    def covmatrix(self):
        """ Get covariance matrix, or None if none present"""
        return self.like.data_covmatrix


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
        tup : tuple
          Mean, upper uncertainty, lower uncertainty.
        """

        pcnt = float(percentile)
        if pcnt < 0 or pcnt > 100:
            raise ValueError("Invalid percentile %f" % pcnt)

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
                raise Exception("No elements survive lower/upper limit clipping")
            if len(cond) != len(aint):
                aint = aint[cond]

        aint.sort()
        mnval = np.mean(aint)
        pval = (1.0 - 0.01 * pcnt) / 2
        na = len(aint)
        lowidx = int(round(pval * na))
        assert lowidx > 0 and lowidx < na, \
            "Invalid lower index %d for pval %f percentile %f" %\
            (lowidx, pval, pcnt)
        lowval = aint[lowidx]
        upidx = int(round((1.0 - pval) * na))
        assert upidx > 0 and upidx < na, \
            "Invalid upper index %d for pval %f percentile %f" %\
            (upidx, pval, pcnt)
        upval  = aint[upidx]
        return (mnval, upval-mnval, mnval-lowval)

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

        if isinstance(param, str):
            paridx = self._param_order[param.lower()]
        else:
            paridx = int(param)
            if paridx < 0 or paridx > 5:
                raise ValueError("invalid parameter index %d" % paridx)

        return self.chain[:,:,paridx].flatten()

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
        tup : tuple
          A tuple of the mean value, upper confidence limit,
          and lower confidence limit.
          Percentile of limit to compute
        """

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

        if isinstance(param, str):
            paridx = self._param_order[param.lower()]
        else:
            paridx = int(param)
            if paridx < 0 or paridx > 5:
                raise ValueError("invalid parameter index %d" % paridx)

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

        if isinstance(param, str):
            paridx = self._param_order[param.lower()]
        else:
            paridx = int(param)
            if paridx < 0 or paridx > 5:
                raise ValueError("invalid parameter index %d" % paridx)

        if percentile <= 0 or percentile >= 100.0:
            raise ValueError("percentile needs to be between 0 and 100")

        svals = self.chain[:,:,paridx].flatten()
        svals.sort()
        return svals[round(0.01 * percentile * len(svals))]

    @property
    def has_peaklambda(self):
        return self._has_peaklambda

    @property
    def peaklambda_chain(self):
        """ Get flattened chain of peak lambda values in microns"""
        
        if not self._has_peaklambda: return None
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
        tup : tuple
          A tuple of the central value, upper uncertainty, 
          and lower uncertainty of the peak observer frame
          wavelength in microns.
        """
        
        if not self._has_peaklambda: return None
        return self._parcen_internal(self.peaklambda.flatten(), percentile,
                                     lowlim=lowlim, uplim=uplim)

    def compute_peaklambda(self):
        """ Compute the observer frame wavelength of peak emission in microns from chain"""

        shp = self.chain.shape[0:2]
        self.peaklambda = np.empty(shp, dtype=np.float)
        for walkidx in range(shp[0]):
            # Do first step
            prevstep = self.chain[walkidx,0,:]
            sed = modified_blackbody(prevstep[0], prevstep[1], prevstep[2],
                                     prevstep[3], prevstep[4], 
                                     opthin=self.opthin,
                                     noalpha=self.noalpha)
            self.peaklambda[walkidx, 0] = sed.max_wave()
            
            #Now other steps
            for stepidx in range(1, shp[1]):
                currstep = self.chain[walkidx,stepidx,:]
                if np.allclose(prevstep, currstep):
                    # Repeat, so avoid re-computation
                    self.peaklambda[walkidx, stepidx] = \
                        self.peaklambda[walkidx, stepidx-1]
                else:
                    sed = modified_blackbody(currstep[0], currstep[1], 
                                             currstep[2], currstep[3], 
                                             currstep[4], 
                                             opthin=self.opthin,
                                             noalpha=self.noalpha)
                    self.peaklambda[walkidx, stepidx] =\
                        sed.max_wave()
                    prevstep = currstep

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
        
        if not self._has_lir: return None
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
        tup : tuple
          A tuple of the central value, upper uncertainty, 
          and lower uncertainty of the IR luminosity (8-1000um)
          in 10^12 solar luminosities, or None if the L_IR has
          not been computed
        """
        if not self._has_lir: return None
        return self._parcen_internal(self.lir.flatten(), percentile,
                                     lowlim=lowlim, uplim=uplim)

    def compute_lir(self, wavemin=8.0, wavemax=1000.0,
                    maxidx=None, lumdist=None):
        """ Computes LIR from chain in 10^12 L_sun.

        Parameters
        ----------
        wavemin : float
          Minimum wavelength of L_IR integration, in microns

        wavemax : float
          Maximum wavelength of L_IR integration, in microns

        maxidx : int
          Maximum index in each walker to use. 

        lumdist : float
          Luminosity distance in Mpc.  Otherwise computed from redshift
          assuming WMAP 9 cosmological model.

        Notes
        -----
        The traditional definition of L_IR is 8-1000um.  Sometimes
        42.5-112.5um is used.
        """

        if self._z is None:
            raise Exception("Redshift must be set to compute L_IR")

        self._lir_min = float(wavemin)
        self._lir_max = float(wavemax)
        if self._lir_min <= 0:
            raise ValueError("Invalid wavemin: %f" % self._lir_min)
        if self._lir_max <= 0:
            raise ValueError("Invalid wavemax: %f" % self._lir_max)
        # Get them in ascending order
        if self._lir_min > self._lir_max:
            self._lir_min, self._lir_max = self._lir_max, self._lir_min

        # 4*pi*dl^2/L_sun in cgs -- so the output will be in 
        # solar luminosities; the prefactor is
        # 4 * pi * mpc_to_cm^2/L_sun
        if not lumdist is None:
            if self._z <= -1:
                raise ValueError("Redshift is less than -1: %f" % self._z)
            dl = float(lumdist)
            if dl <= 0.0:
                raise ValueError("Invalid luminosity distance: %f" % dl)
        else:
            if self._z <= 0:
                raise ValueError("Redshift is invalid: %f" % self._z)
            import astropy.cosmology
            dl = astropy.cosmology.WMAP9.luminosity_distance(self._z) #Mpc

        lirprefac = 3.11749657e4 * dl**2 # Also converts to 10^12 lsolar

        # L_IR defined as between 8 and 1000 microns (rest)
        integrator = mbb_freqint(self._z, self._lir_min, self._lir_max,
                                 opthin=self.opthin,
                                 noalpha=self.noalpha)

        # Now we compute L_IR for every step taken.
        # Explicitly check for repeats
        shp = self.chain.shape[0:2]
        steps = shp[1]
        if not maxidx is None:
            if maxidx < steps: steps = maxidx
        self.lir = np.empty((shp[0],steps), dtype=np.float)
        for walkidx in range(shp[0]):
            # Do first step
            prevstep = self.chain[walkidx,0,:]
            self.lir[walkidx,0] = \
                lirprefac * integrator(prevstep)
            for stepidx in range(1, steps):
                currstep = self.chain[walkidx,stepidx,:]
                if np.allclose(prevstep, currstep):
                    # Repeat, so avoid re-computation
                    self.lir[walkidx, stepidx] =\
                        self.lir[walkidx, stepidx-1]
                else:
                    # New step
                    self.lir[walkidx, stepidx] = \
                        lirprefac * integrator(currstep)
                    prevstep = currstep

        self._has_lir = True

    @property
    def has_dustmass(self):
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

        if not self._has_dustmass: return None
        return self.dustmass.flatten()

    def dustmass_cen(self, percentile=68.3, lowlim=None, uplim=None):
        """ Gets the central confidence interval for dustmass.

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
        tup : tuple
          A tuple of the central value, upper uncertainty, 
          and lower uncertainty of the dust mass in 10^8 solar masses,
          or None if not computed.
        """

        if not self._has_dustmass: return None
        return self._parcen_internal(self.dustmass.flatten(), percentile,
                                     lowlim=lowlim, uplim=uplim)


    def _dmass_calc(self, step, opz, bnu_fac, temp_fac, knu_fac,
                    opthin, dl2):
        """Internal function to comput dustmass in 10^8 M_sun, 
        given various pre-computed values"""

        msolar8 = 1.97792e41 ## mass of the sun*10^8 in g
        T = step[0] * opz
        beta = step[1]
        S_nu = step[4] * 1e-26 # to erg / s-cm^2-Hz from mJy
        B_nu = bnu_fac / math.expm1(temp_fac / T) #Planck function
        #Scale kappa with freq (obs frame ok).  Factor of 10 is
        # m^2 kg^-1 -> cm^2 g^-1 conversion
        K_nu = 10.0 * self._kappa * knu_fac**(-beta) 
        dustmass = dl2 * S_nu / (opz * K_nu * B_nu * msolar8)
        if not opthin:
            tau_nu = (step[2] / self.like.wavenorm)**beta
            op_fac = - tau_nu / math.expm1(-tau_nu)
            dustmass *= op_fac
        return dustmass

    def compute_dustmass(self, kappa=2.64, kappa_wave=125.0,
                         maxidx=None, lumdist=None):
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

        lumdist : float
          Luminosity distance in Mpc.  Otherwise computed from redshift
          assuming WMAP 9 cosmological model.
        """

        if self._z is None:
            raise Exception("Redshift must be set to compute dust mass")

        self._kappa = float(kappa)
        self._kappa_wave = float(kappa_wave)

        if self._kappa <= 0:
            raise ValueError("Invalid (non-positive) kappa %f" % self._kappa)
        if self._kappa_wave <= 0:
            raise ValueError("Invalid (non-positive) kappa wavelength %f" % \
                                 self._kappa_wave)
        
        # Get luminosity distance
        if not lumdist is None:
            if self._z <= -1:
                raise ValueError("Redshift is less than -1: %f" % self._z)
            dl = float(lumdist)
            if dl <= 0.0:
                raise ValueError("Invalid luminosity distance: %f" % dl)
        else:
            if self._z <= 0:
                raise ValueError("Redshift is invalid: %f" % self._z)
            import astropy.cosmology
            dl = astropy.cosmology.WMAP9.luminosity_distance(self._z) #Mpc

        mpc_to_cm = 3.08567758e24
        dl *= mpc_to_cm
        dl2 = dl**2
        opz = 1.0 + self._z

        wavenorm_rest = self.wavenorm / opz # in um
        nunorm_rest = 299792458e6 / wavenorm_rest # in Hz

        # Precompute some quantities for evaluating the Planck function
        # h nu / k and 2 h nu^3 / c^2
        temp_fac = 6.6260693e-27 * nunorm_rest / 1.38065e-16  #h nu / k
        bnu_fac = 2 * 6.6260693e-27 * nunorm_rest**3 / 299792458e2**2

        # Work out dust opacity factor. 
        knu_fac = wavenorm_rest / self._kappa_wave

        msolar8 = 1.97792e41 ## mass of the sun*10^8 in g

        shp = self.chain.shape[0:2]
        steps = shp[1]
        if not maxidx is None:
            if maxidx < steps: steps = maxidx
        self.dustmass = np.empty((shp[0],steps), dtype=np.float)
        for walkidx in range(shp[0]):
            # Do first step
            prevstep = self.chain[walkidx,0,:]
            self.dustmass[walkidx,0] = \
                self._dmass_calc(prevstep, opz, bnu_fac, temp_fac, knu_fac, 
                                 self.opthin, dl2)
            for stepidx in range(1, steps):
                currstep = self.chain[walkidx,0,:]
                if np.allclose(prevstep, currstep):
                    # Repeat, so avoid re-computation
                    self.dustmass[walkidx, stepidx] = \
                        self.dustmass[walkidx, stepidx-1]
                else:
                    self.dustmass[walkidx, stepidx] = \
                        self._dmass_calc(currstep, opz, bnu_fac,
                                         temp_fac, knu_fac, 
                                         self.opthin, dl2)
                    prevstep = currstep

        self._has_dustmass = True

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
            if not self.like._response_integrate:
                raise Error("Asked for response integration, but no "
                            " response functions available from original fit")
            if not self.like.has_response(spec):
                raise ValueError("Do not have response function matching %s" % spec)
            resp = self.like.get_response(spec)
            doing_response = True
        else:
            wv = float(spec)
            if (wv <= 0):
                raise ValueError("Invalid wavelength %f" % wv)
            doing_response = False

        # Do computation.  explicitly checking for repeats.
        shp = self.chain.shape[0:2]
        steps = shp[1]
        if not maxidx is None:
            if maxidx < steps: steps = maxidx
        fpred = np.empty((shp[0],steps), dtype=np.float)
        for walkidx in range(shp[0]):
            # Do first step
            prevstep = self.chain[walkidx,0,:]
            sed = modified_blackbody(prevstep[0], prevstep[1], prevstep[2],
                                     prevstep[3], prevstep[4], 
                                     opthin=self.opthin,
                                     noalpha=self.noalpha)
            if doing_response:
                fpred[walkidx, 0] = resp(sed)
            else:
                fpred[walkidx, 0] = sed(wv)

            for stepidx in range(1, steps):
                currstep = self.chain[walkidx,stepidx,:]
                if np.allclose(prevstep, currstep):
                    # Repeat, so avoid re-computation
                    fpred[walkidx, stepidx] =\
                        fpred[walkidx, stepidx-1]
                else:
                    sed = modified_blackbody(currstep[0], currstep[1], 
                                             currstep[2], currstep[3], 
                                             currstep[4], opthin=self.opthin,
                                             noalpha=self.noalpha)
                    if doing_response:
                        fpred[walkidx, stepidx] = resp(sed)
                    else:
                        fpred[walkidx, stepidx] = sed(wv)

                    prevstep = currstep

        return fpred

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
        tup : tuple
          A tuple of the central value, upper uncertainty, 
          and lower uncertainty of the predicted flux in mJy.
          If spec was a float, this is the sed flux at that value.  
          If it was a string, it is the response predicted for that 
          response function name. So, for example, if spec = 'SPIRE_250um' 
          it will be the predicted flux integrated through the Herschel-SPIRE
          250um filter function -- if that was available to the fit.
        """

        pflux = self._predict_flux(spec, maxidx)

        return self._parcen_internal(pflux.flatten(), percentile,
                                     lowlim=lowlim, uplim=uplim)


    def __str__(self):
        """ String representation of results"""
        idx = [0,1,4]
        tag = ["T/(1+z)","beta","fnorm"]
        units = ["[K]","","[mJy]"]
        retstr = ""
        
        for i,tg, unit in zip(idx, tag, units):
            retstr += "%s: " % tg
            if self.fixed[i]:
                retstr += "%0.2f (fixed)\n" % self.chain[:,:,i].mean()
            else:
                retstr += "%0.2f +%0.2f -%0.2f" % self.par_central_values[i]
                retstr += " (low lim: %0.2f" % self.like.lowlim(i)
                if self.like.has_uplim(i):
                    retstr += " upper lim: %0.2f" % self.like.uplim(i)
                if self.like.has_gaussian_prior(i):
                    retstr += " prior: %0.2f %0.2f" %\
                        self.like.get_gaussian_prior(i)
                retstr += ") %s\n" % unit

        if not self.opthin:
            if self.fixed[2]:
                retstr += "lambda0 (1+z): %0.2f (fixed) [um]\n" %\
                    self.chain[:,:,2].mean()
            else:
                retstr += "lambda0 (1+z): %0.2f +%0.2f -%0.2f" %\
                    self.par_central_values[2]
                retstr += " (low lim: %0.2f" % self.like.lowlim(2)
                if self.like.has_uplim(2):
                    retstr += " upper lim: %0.2f" % self.like.uplim(2)
                if self.like.has_gaussian_prior(2):
                    retstr += " prior: %0.2f %0.2f" %\
                        self.like.get_gaussian_prior(2)
                retstr += ") [um]\n"
        else:
            retstr += "Optically thin case assumed\n"

        if not self.like.noalpha:
            if self.fixed[3]:
                retstr += "alpha: %0.2f (fixed)\n" % self.chain[:,:,3].mean()
            else:
                retstr += "alpha: %0.2f +%0.2f -%0.2f" %\
                    self.par_central_values[3]
                retstr += " (low lim: %0.2f" % self.like.lowlim(3)
                if self.like.has_uplim(3):
                    retstr += " upper lim: %0.2f" % self.like.uplim(3)
                if self.like.has_gaussian_prior(3):
                    retstr += " prior: %0.2f %0.2f" %\
                        self.like.get_gaussian_prior(3)
                retstr += ")\n"
        else:
            retstr += "Alpha not used\n"
        
        retstr += "Number of data points: %d\n" % self.like.ndata
        retstr += "ChiSquare of best fit point: %0.2f" % self.best_fit_chisq

        return retstr

############################################################

# The idea is to allow this to be multiprocessed
class mbb_freqint(object):
    """ Does frequency integration"""
    
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
            raise Exception("Invalid (negative) redshift: %f" % self._redshift)
        if self._lammin <= 0:
            raise Exception("Invalid (non-positive) lammin: %f" % self._lammin)
        if self._lammax <= 0:
            raise Exception("Invalid (non-positive) lammax: %f" % self._lammax)
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

############################################################

class mbb_fit(object):
    """ Does fit"""

    """ Parameter order dictionary.  Lowercased."""
    _param_order = {'t': 0, 't/(1+z)': 0, 'beta': 1, 'lambda0': 2,
                    'lambda0*(1+z)': 2, 'lambda_0': 2, 'lambda_0*(1+z)': 2,
                    'alpha': 3, 'fnorm': 4, 'f500': 4}

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
          parameter.
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
          parameter.

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
          parameter.
          
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
          parameter.
          
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
        This will take care of fixed parameters correctly.
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
