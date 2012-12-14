import numpy
import emcee
import math
import multiprocessing
from modified_blackbody import modified_blackbody
from likelihood import likelihood
import copy

__all__ = ["mbb_fit", "mbb_fit_results"]

# This class holds the results.  Why don't we just save the
# fit class?  Because it can involve a multiprocessing pool,
# which can't be pickled.  So instead we package the results up
# in this, which also has methods for finding central limits, etc.
class mbb_fit_results(object):
    """Holds results of fit"""
    def __init__(self, fit):
        """
        Parameters
        ----------
        fit : mbb_fit
          Fit object
        """

        assert type(fit) is mbb_fit, "fit is not mbb_fit"

        self.like = fit.like
        self.chain = fit.sampler.chain
        self.lnprobability = fit.sampler.lnprobability

        self.par_central_values = [self.par_cen(i) for i in range(5)]

        try:
            self.lir = fit.lir
            self.lir_central_value = self.lir_cen()
        except AttributeError:
            pass
        try:
            self.lagn = fit.lagn
            self.lagn_central_value = self.lagn_cen()
        except AttributeError:
            pass
        try:
            self.dustmass = fit.dustmass
            self.dustmass_central_value = self.dustmass_cen()
        except AttributeError:
            pass
        try:
            self.peaklambda = fit.peaklambda
            self.peaklambda_central_value = self.peaklambda_cen()
        except AttributeError:
            pass

    def best_fit(self):
        """ Finds the best fitting point that occurred during the fit

        Returns
        -------
        tup : tuple
         A tuple of the parameters, the log probability, and the
         index into lnprobability"""

        idxmax_flat = self.lnprobability.argmax()
        idxmax = numpy.unravel_index(idxmax_flat, self.lnprobability.shape)
        return (self.chain[idxmax[0], idxmax[1], :],
                self.lnprobability[idxmax[0], idxmax[1]],
                idxmax)

    def _parcen_internal(self, array, percentile, lowlim=None,
                         uplim=None):
        """
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
                cond = numpy.all(aint >= float(lowlim),
                                 aint <= float(uplim)).nonzero()[0]
            if len(cond) == 0:
                raise Exception("No elements survive lower/upper limit clipping")
            if len(cond) != len(aint):
                aint = aint[cond]

        aint.sort()
        mnval = numpy.mean(aint)
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

        if not hasattr(self, 'peaklambda'): return None
        return self._parcen_internal(self.peaklambda.flatten(), percentile,
                                     lowlim=lowlim, uplim=uplim)

    @property
    def lir_chain(self):
        if not hasattr(self, 'lir'): return None
        return self.lir.flatten()

    def lir_cen(self, percentile=68.3, lowlim=None, uplim=None):
        """ Gets the central confidence interval for L_IR.

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
          in 10^12 solar luminosities.
        """

        if not hasattr(self, 'lir'): return None
        return self._parcen_internal(self.lir.flatten(), percentile,
                                     lowlim=lowlim, uplim=uplim)

    @property
    def lagn_chain(self):
        if not hasattr(self, 'lagn'): return None
        return self.lagn.flatten()

    def lagn_cen(self, percentile=68.3, lowlim=None, uplim=None):
        """ Gets the central confidence interval for L_AGN

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
          and lower uncertainty of the luminosity from 42.5-122.5um
          in 10^12 solar luminosities.
        """

        if not hasattr(self, 'lagn'): return None
        return self._parcen_internal(self.lagn.flatten(), percentile,
                                     lowlim=lowlim, uplim=uplim)

    @property
    def dustmass_chain(self):
        if not hasattr(self, 'dustmass'): return None
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
          and lower uncertainty of the dust mass in 10^8 solar masses.
        """

        if not hasattr(self, 'dustmass'): return None
        return self._parcen_internal(self.dustmass.flatten(), percentile,
                                     lowlim=lowlim, uplim=uplim)

    def parameter_chain(self, paridx):
        """ Gets flattened chain for parameter number"""
        if paridx < 0 or paridx > 5:
            raise ValueError("invalid parameter index %d" % paridx)
        return self.chain[:,:,paridx].flatten()

    def par_cen(self, paridx, percentile=68.3, lowlim=None, uplim=None):
        """ Gets the central confidence interval for the parameter

        The parameters are in the order T, beta, lambda0, alpha, fnorm"""

        if percentile <= 0 or percentile >= 100.0:
            raise ValueError("percentile needs to be between 0 and 100")
        if paridx < 0 or paridx > 5:
            raise ValueError("invalid parameter index %d" % paridx)

        return self._parcen_internal(self.parameter_chain(paridx), 
                                     percentile, lowlim=lowlim, uplim=uplim)

    def par_lowlim(self, paridx, percentile=68.3):
        """ Gets the lower limit for the parameter

        The parameters are in the order T, beta, lambda0, alpha, fnorm"""

        if percentile <= 0 or percentile >= 100.0:
            raise ValueError("percentile needs to be between 0 and 100")
        if paridx < 0 or paridx > 5:
            raise ValueError("invalid parameter index %d" % paridx)

        svals = self.parameter_chain(paridx)
        svals.sort()
        return svals[round((1.0 - 0.01 * percentile) * len(svals))]

    def par_uplim(self, paridx, percentile=68.3):
        """ Gets the upper limit for the parameter

        The parameters are in the order T, beta, lambda0, alpha, fnorm"""

        if percentile <= 0 or percentile >= 100.0:
            raise ValueError("percentile needs to be between 0 and 100")
        if paridx < 0 or paridx > 5:
            raise ValueError("invalid parameter index %d" % paridx)

        svals = self.chain[:,:,paridx].flatten()
        svals.sort()
        return svals[round(0.01 * percentile * len(svals))]

    def __repr__(self):
        """ Print out the parameter central values"""
        idx = [0,1,4]
        tag = ["T/(1+z)","beta","fnorm"]
        units = ["[K]","","[mJy]"]
        retstr = ""
        for i,tg, unit in zip(idx, tag, units):
            retstr += "%s: " % tg
            if self.like._fixed[i]:
                retstr += "%0.2f (fixed)\n" % self.chain[:,:,i].mean()
            else:
                retstr += "%0.2f +%0.2f -%0.2f" % self.par_central_values[i]
                retstr += " (low lim: %0.2f" % self.like._lowlim[i]
            if self.like._has_uplim[i]:
                retstr += " upper lim: %0.2f" % self.like._uplim[i]
            if self.like._has_gprior[i]:
                tup = (self.like._gprior_mean[i], 
                       1.0/math.sqrt(self.like._gprior_ivar[i]))
                retstr += " prior: %0.2f %0.2f" % tup
            retstr += ") %s\n" % unit

        if not self.like._opthin:
            if self.like._fixed[2]:
                retstr += "lambda0 (1+z): %0.2f (fixed) [um]\n" %\
                    self.chain[:,:,2].mean()
            else:
                retstr += "lambda0 (1+z): %0.2f +%0.2f -%0.2f" %\
                    self.par_central_values[2]
                retstr += " (low lim: %0.2f" % self.like._lowlim[2]
                if self.like._has_uplim[2]:
                    retstr += " upper lim: %0.2f" % self.like._uplim[2]
                if self.like._has_gprior[2]:
                    tup = (self.like._gprior_mean[2],
                           1.0/math.sqrt(self.like._gprior_ivar[2]))
                    retstr += " prior: %0.2f %0.2f" % tup
                retstr += ") [um]\n"
        else:
            retstr += "Optically thin case assumed\n"

        if not self.like._noalpha:
            if self.like._fixed[3]:
                retstr += "alpha: %0.2f (fixed)\n" % self.chain[:,:,3].mean()
            else:
                retstr += "alpha: %0.2f +%0.2f -%0.2f" % self.par_central_values[3]
                retstr += " (low lim: %0.2f" % self.like._lowlim[3]
                if self.like._has_uplim[3]:
                    retstr += " upper lim: %0.2f" % self.like._uplim[3]
                if self.like._has_gprior[3]:
                    tup = (self.like._gprior_mean[3],
                           1.0/math.sqrt(self.like._gprior_ivar[3]))
                    retstr += " prior: %0.2f %0.2f" % tup
                retstr += ")\n"
        else:
            retstr += "Alpha not used\n"

        if hasattr(self,'lir_central_value'):
            retstr += "L_IR: %0.2f +%0.2f -%0.2f [10^12 Lsun]\n" % \
                self.lir_central_value
        if hasattr(self,'lagn_central_value'):
            retstr += "L_AGN: %0.2f +%0.2f -%0.2f [10^12 Lsun]\n" % \
                self.lagn_central_value
        if hasattr(self,'dustmass_central_value'):
            retstr += "M_dust: %0.2f +%0.2f -%0.2f [10^8 Msun]\n" % \
                self.dustmass_central_value
        if hasattr(self,'peaklambda_central_value'):
            retstr += "lambda_peak: %0.2f +%0.2f -%0.2f [um]\n" % \
                self.peaklambda_central_value
            
        return retstr

# The idea is to allow this to also be multiprocessed
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
        """ Evaluates frequency integral."""
        mbb = modified_blackbody(params[0], params[1], params[2],
                                 params[3], params[4], opthin=self._opthin,
                                 noalpha=self._noalpha)
        return mbb.freq_integrate(self._minwave_obs, self._maxwave_obs)


class mbb_fit(object):
    """ Does fit"""

    def __init__(self, photfile, covfile, covextn, wavenorm, noalpha, 
                 opthin, nwalkers, nthreads):
        """
        Parameters
        __________
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

        nwalkers : integer
           Number of MCMC walkers to use in fit

        nthreads : integer
           Number of threads to use
        """

        self._noalpha = noalpha
        self._opthin = opthin
        self._wavenorm = float(wavenorm)
        self._nthreads = int(nthreads)
        self.like = likelihood(photfile, covfile=covfile, covextn=covextn, 
                               wavenorm=wavenorm, noalpha=noalpha, 
                               opthin=opthin)
        self.sampler = emcee.EnsembleSampler(nwalkers, 5, self.like,
                                             threads=self._nthreads)
        self._sampled = False

    def run(self, nburn, nsteps, p0, verbose=False):
        """Do emcee run"""

        # Do burn in
        self.sampler.reset()
        self._sampled = False

        if nburn <= 0:
            errmsg = "Invalid (non-positive) number of burn in steps: %d"
            raise ValueError(errmsg % nburn)
        if verbose:
            print "Doing burn in with %d steps" % nburn
        pos, prob, rstate = self.sampler.run_mcmc(p0, nburn)

        # Reset and do main fit
        self.sampler.reset()
        if nsteps <= 0:
            errmsg = "Invalid (non-positive) number of main chain steps: %d"
            raise ValueError(errmsg % nsteps)
        if verbose:
            print "Doing main chain with %d steps" % nsteps
        st = self.sampler.run_mcmc(pos, nsteps, rstate0=rstate)
        self._sampled = True

        if verbose:
            print "Fit complete"
            print " Mean acceptance fraction:", \
                numpy.mean(self.sampler.acceptance_fraction)
            try :
                acor = self.sampler.acor
                print " Autocorrelation time: "
                print "  Number of burn in steps (%d) should be larger than these" % \
                    nburn
                print "\tT:        %f" % acor[0]
                print "\tbeta:     %f" % acor[1]
                if not self._opthin:
                    print "\tlambda0:  %f" % acor[2]
                if not self._noalpha:
                    print "\talpha:    %f" % acor[3]
                print "\tfnorm:    %f" % acor[4]
            except ImportError :
                pass

    def get_peaklambda(self):
        """ Find the wavelength of peak emission in microns from chain"""

        shp = self.sampler.chain.shape[0:2]
        self.peaklambda = numpy.empty(shp, dtype=numpy.float)
        for walkidx in range(shp[0]):
            # Do first step
            prevstep = self.sampler.chain[walkidx,0,:]
            sed = modified_blackbody(prevstep[0], prevstep[1], prevstep[2],
                                     prevstep[3], prevstep[4], 
                                     opthin=self._opthin,
                                     noalpha=self._noalpha)
            self.peaklambda[walkidx, 0] = sed.max_wave()
            
            #Now other steps
            for stepidx in range(1, shp[1]):
                currstep = self.sampler.chain[walkidx,stepidx,:]
                if numpy.allclose(prevstep, currstep):
                    # Repeat, so avoid re-computation
                    self.peaklambda[walkidx, stepidx] = \
                        self.peaklambda[walkidx, stepidx-1]
                else:
                    sed = modified_blackbody(currstep[0], currstep[1], 
                                             currstep[2], currstep[3], 
                                             currstep[4], 
                                             opthin=self._opthin,
                                             noalpha=self._noalpha)
                    self.peaklambda[walkidx, stepidx] =\
                        sed.max_wave()
                    prevstep = currstep

    def get_lir(self, redshift, maxidx=None):
        """ Get 8-1000 micron LIR from chain in 10^12 solar luminosities"""

        try:
            import astropy.cosmology
        except ImportError:
            raise ImportError("Need to have astropy installed if getting LIR")

        if not self._sampled:
            raise Exception("Chain has not been run in get_lir")

        # 4*pi*dl^2/L_sun in cgs -- so the output will be in 
        # solar luminosities; the prefactor is
        # 4 * pi * mpc_to_cm^2/L_sun
        z = float(redshift)
        if z <= 0:
            raise ValueError("Redshift is not positive: %f" % z)
        dl = astropy.cosmology.WMAP7.luminosity_distance(z) #Mpc
        lirprefac = 3.11749657e4 * dl**2 # Also converts to 10^12 lsolar

        # L_IR defined as between 8 and 1000 microns (rest)
        integrator = mbb_freqint(z, 8.0, 1000.0, opthin=self._opthin,
                                 noalpha=self._noalpha)

        # Now we compute L_IR for every step taken.
        # Two cases: using multiprocessing, and serially.
        if self._nthreads > 1:
            shp = self.sampler.chain.shape[0:2]
            npar = self.sampler.chain.shape[2]
            nel = shp[0] * shp[1]
            pool = multiprocessing.Pool(self._nthreads)
            rchain = self.sampler.chain.reshape(nel, npar)
            lir = numpy.array(pool.map(integrator,
                                       [rchain[i] for i in xrange(nel)]))
            self.lir = lirprefac * lir.reshape((shp[0], shp[1]))
        else :
            # Explicitly check for repeats
            shp = self.sampler.chain.shape[0:2]
            steps = shp[1]
            if not maxidx is None:
                if maxidx < steps: steps = maxidx
            self.lir = numpy.empty((shp[0],steps), dtype=numpy.float)
            for walkidx in range(shp[0]):
                # Do first step
                prevstep = self.sampler.chain[walkidx,0,:]
                self.lir[walkidx,0] = \
                    lirprefac * integrator(prevstep)
                for stepidx in range(1, steps):
                    currstep = self.sampler.chain[walkidx,stepidx,:]
                    if numpy.allclose(prevstep, currstep):
                        # Repeat, so avoid re-computation
                        self.lir[walkidx, stepidx] =\
                            self.lir[walkidx, stepidx-1]
                    else:
                        self.lir[walkidx, stepidx] = \
                            lirprefac * integrator(prevstep)
                        prevstep = currstep

    def get_lagn(self, redshift, maxidx=None):
        """Get 42.5-112.5 micron luminosity from chain in 10^12 solar 
        luminosites"""

        try:
            import astropy.cosmology
        except ImportError:
            raise ImportError("Need to have astropy installed if getting LAGN")
        
        if not self._sampled:
            raise Exception("Chain has not been run in get_agn")

        # Get luminosity distance in cm for correction
        z = float(redshift)
        if z <= 0:
            raise ValueError("Redshift is not positive: %f" % z)
        dl = astropy.cosmology.WMAP7.luminosity_distance(z)

        # 4*pi*dl^2/L_sun in cgs -- so the output will be in 
        # solar luminosities; the prefactor is
        # 4 * pi * mpc_to_cm^2/L_sun
        lagnprefac = 3.11749657e4 * dl**2

        # L_IR defined as between 42.5 and 122.5 microns (rest)
        integrator = mbb_freqint(z, 42.5, 122.5, opthin=self._opthin,
                                 noalpha=self._noalpha)

        # Now we compute L_AGN for every step taken.
        # Two cases: using multiprocessing, and serially.
        if self._nthreads > 1:
            shp = self.sampler.chain.shape[0:2]
            npar = self.sampler.chain.shape[2]
            nel = shp[0] * shp[1]
            pool = multiprocessing.Pool(self._nthreads)
            rchain = self.sampler.chain.reshape(nel, npar)
            lagn = numpy.array(pool.map(integrator,
                                        [rchain[i] for i in xrange(nel)]))
            self.lagn = lagnprefac * lagn.reshape(shp[0], shp[1])
        else :
            # Explicitly check for repeats
            shp = self.sampler.chain.shape[0:2]
            steps = shp[1]
            if not maxidx is None:
                if maxidx < steps: steps = maxidx
            self.lagn = numpy.empty((shp[0],steps), dtype=numpy.float)
            for walkidx in range(shp[0]):
                # Do first step
                prevstep = self.sampler.chain[walkidx,0,:]
                self.lagn[walkidx,0] = \
                    lagnprefac * integrator(prevstep)
                for stepidx in range(1, steps):
                    currstep = self.sampler.chain[walkidx,stepidx,:]
                    if numpy.allclose(prevstep, currstep):
                        # Repeat, so avoid re-computation
                        self.lagn[walkidx, stepidx] =\
                            self.lagn[walkidx, stepidx-1]
                    else:
                        self.lagn[walkidx, stepidx] = \
                            lagnprefac * integrator(prevstep)
                        prevstep = currstep


    def _dmass_calc(self, step, opz, bnu_fac, temp_fac, knu_fac,
                    opthin, dl2):
        """Internal function to comput dustmass in 10^8 M_sun, 
        given various pre-computed values"""

        msolar8 = 1.97792e41 ## mass of the sun*10^8 in g
        T = step[0] * opz
        beta = step[1]
        S_nu = step[4] * 1e-26 # to erg / s-cm^2-Hz from mJy
        B_nu = bnu_fac / math.expm1(temp_fac / T) #Planck function
        # Dunne optical depth is 2.64 m^2 kg^-1 = 26.4 cm^2 g^-1
        K_nu = 26.4 * knu_fac**(-beta) #Scaling with freq (obs frame ok)
        dustmass = dl2 * S_nu / (opz * K_nu * B_nu * msolar8)
        if not opthin:
            tau_nu = (step[2] / self._wavenorm)**beta
            op_fac = - tau_nu / math.expm1(-tau_nu)
            dustmass *= op_fac
        return dustmass

    def get_dustmass(self, redshift, maxidx=None):
        # This one is not parallelized because the calculation
        # is relatively trivial
        """Get dust mass in 10^8 M_sun from chain"""

        try:
            import astropy.cosmology
        except ImportError:
            raise ImportError("Need to have astropy installed if getting LIR")

        if not self._sampled:
            raise Exception("Chain has not been run in get_mdust")

        # Get luminosity distance
        z = float(redshift)
        if z <= 0:
            raise ValueError("Redshift is not positive: %f" % z)
        mpc_to_cm = 3.08567758e24
        dl = astropy.cosmology.WMAP7.luminosity_distance(z) * mpc_to_cm
        dl2 = dl**2
        opz = 1.0 + z

        wavenorm_rest = self._wavenorm / opz # in um
        nunorm_rest = 299792458e6 / wavenorm_rest # in Hz

        # Precompute some quantities for evaluating the Planck function
        # h nu / k and 2 h nu^3 / c^2
        temp_fac = 6.6260693e-27 * nunorm_rest / 1.38065e-16  #h nu / k
        bnu_fac = 2 * 6.6260693e-27 * nunorm_rest**3 / 299792458e2**2

        # The dust factor we use is defined at 125 microns, rest,
        # and scales with beta
        knu_fac = wavenorm_rest / 125.0

        msolar8 = 1.97792e41 ## mass of the sun*10^8 in g

        shp = self.sampler.chain.shape[0:2]
        steps = shp[1]
        if not maxidx is None:
            if maxidx < steps: steps = maxidx
        self.dustmass = numpy.empty((shp[0],steps), dtype=numpy.float)
        for walkidx in range(shp[0]):
            # Do first step
            prevstep = self.sampler.chain[walkidx,0,:]
            self.dustmass[walkidx,0] = self._dmass_calc(prevstep, opz, bnu_fac,
                                                        temp_fac, knu_fac, 
                                                        self._opthin, dl2)
            for stepidx in range(1, steps):
                currstep = self.sampler.chain[walkidx,0,:]
                if numpy.allclose(prevstep, currstep):
                    # Repeat, so avoid re-computation
                    self.dustmass[walkidx, stepidx] = \
                        self.dustmass[walkidx, stepidx-1]
                else:
                    self.dustmass[walkidx, stepidx] = \
                        self._dmass_calc(currstep, opz, bnu_fac,
                                         temp_fac, knu_fac, 
                                         self._opthin, dl2)
                    prevstep = currstep

