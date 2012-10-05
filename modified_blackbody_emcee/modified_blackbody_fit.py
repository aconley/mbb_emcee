import numpy
import emcee
import math
from modified_blackbody import modified_blackbody
from likelihood import likelihood

__all__ = ["modified_blackbody_fit"]

#Set up class to hold results
class modified_blackbody_fit(object):
    """Does fit"""
    def __init__(self, photfile, covfile, covextn, wavenorm, noalpha, 
                 opthin, nwalkers, npar, threads):
        self._noalpha = noalpha
        self._opthin = opthin
        self._wavenorm = float(wavenorm)
        self.like = likelihood(photfile, covfile=covfile, covextn=covextn, 
                               wavenorm=wavenorm, noalpha=noalpha, 
                               opthin=opthin)
        self.sampler = emcee.EnsembleSampler(nwalkers, npar, self.like,
                                             threads=threads)
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
                print "  Number of steps (%d) should be larger than these" % \
                    nsteps
                print "\tT:        %f" % acor[0]
                print "\tbeta:     %f" % acor[1]
                if not self._opthin:
                    print "\tlambda0:  %f" % acor[2]
                if not self._noalpha:
                    print "\talpha:    %f" % acor[3]
                print "\tfnorm:    %f" % acor[4]
            except ImportError :
                pass

    def get_lir(self, redshift):
        """Get 8-1000 micron LIR from chain"""
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
        dl = astropy.cosmology.WMAP7.luminosity_distance(z) #Mpc
        lirprefac = 3.11749657e16 * dl**2
        opz = 1.0 + z

        # L_IR defined as between 8 and 1000 microns (rest)
        minwave = 8.0 * opz
        maxwave = 1000 * opz

        # Now we compute L_IR for every step taken, checking
        # for repeats
        shp = self.sampler.chain.shape[0:2]
        self.lir = numpy.empty(shp, dtype=numpy.float)
        for walkidx in range(shp[0]):
            # Do first step
            prevstep = self.sampler.chain[walkidx,0,:]
            sed = modified_blackbody(prevstep[0], prevstep[1], prevstep[2],
                                     prevstep[3], prevstep[4], 
                                     opthin=self._opthin,
                                     noalpha=self._noalpha)
            self.lir[walkidx,0] = \
                lirprefac * sed.freq_integrate(minwave, maxwave)
            for stepidx in range(1, shp[1]):
                currstep = self.sampler.chain[walkidx,0,:]
                if numpy.allclose(prevstep, currstep):
                    # Repeat, so avoid re-computation
                    self.lir[walkidx, stepidx] = self.lir[walkidx, stepidx-1]
                else:
                    sed = modified_blackbody(currstep[0], currstep[1], 
                                             currstep[2], currstep[3], 
                                             currstep[4], 
                                             opthin=self._opthin,
                                             noalpha=self._noalpha)
                    self.lir[walkidx, stepidx] = \
                        lirprefac * sed.freq_integrate(minwave, maxwave)
                    prevstep = currstep

    def get_lagn(self, redshift):
        """Get 42.5-112.5 micron luminosity from chain"""
        try:
            import astropy.cosmology
        except ImportError:
            raise ImportError("Need to have astropy installed if getting LAGN")
        
        if not self._sampled:
            raise Exception("Chain has not been run in get_agn")

        # Get luminosity distance in cm for correction
        z = float(redshift)
        dl = astropy.cosmology.WMAP7.luminosity_distance(z)

        # 4*pi*dl^2/L_sun in cgs -- so the output will be in 
        # solar luminosities; the prefactor is
        # 4 * pi * mpc_to_cm^2/L_sun
        lagnprefac = 3.11749657e16 * dl**2
        opz = 1.0 + z

        # L_IR defined as between 8 and 1000 microns (rest)
        minwave = 42.5 * opz
        maxwave = 122.5 * opz

        # Now we compute L_IR for every step taken, checking
        # for repeats
        shp = self.sampler.chain.shape[0:2]
        self.lagn = numpy.empty(shp, dtype=numpy.float)
        for walkidx in range(shp[0]):
            # Do first step
            prevstep = self.sampler.chain[walkidx,0,:]
            sed = modified_blackbody(prevstep[0], prevstep[1], prevstep[2],
                                     prevstep[3], prevstep[4], 
                                     opthin=self._opthin,
                                     noalpha=self._noalpha)
            self.lagn[walkidx,0] = \
                lagnprefac * sed.freq_integrate(minwave, maxwave)
            for stepidx in range(1, shp[1]):
                currstep = self.sampler.chain[walkidx,0,:]
                if numpy.allclose(prevstep, currstep):
                    # Repeat, so avoid re-computation
                    self.lagn[walkidx, stepidx] = self.lagn[walkidx, stepidx-1]
                else:
                    sed = modified_blackbody(currstep[0], currstep[1], 
                                             currstep[2], currstep[3], 
                                             currstep[4], 
                                             opthin=self._opthin,
                                             noalpha=self._noalpha)
                    self.lagn[walkidx, stepidx] = \
                        lagnprefac * sed.freq_integrate(minwave, maxwave)
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

    def get_dustmass(self, redshift):
        """Get dust mass in 10^8 M_sun from chain"""
        try:
            import astropy.cosmology
        except ImportError:
            raise ImportError("Need to have astropy installed if getting LIR")

        if not self._sampled:
            raise Exception("Chain has not been run in get_mdust")

        # Get luminosity distance
        z = float(redshift)
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
        self.dustmass = numpy.empty(shp, dtype=numpy.float)
        for walkidx in range(shp[0]):
            # Do first step
            prevstep = self.sampler.chain[walkidx,0,:]
            self.dustmass[walkidx,0] = self._dmass_calc(prevstep, opz, bnu_fac,
                                                        temp_fac, knu_fac, 
                                                        self._opthin, dl2)
            for stepidx in range(1, shp[1]):
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

