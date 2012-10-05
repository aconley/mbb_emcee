import math
import numpy
import scipy.optimize
from scipy.special import lambertw

from utility import isiterable

"""Modified blackbody SED"""

__all__ = ["modified_blackbody"]

def alpha_merge_eqn(x, alpha, beta, x0, opthin=False):
    """Equation we need the root for to merge power law to modified
    blackbody"""
    try :
        # This can overflow badly
        xox0beta = (x / x0)**beta
        bterm = xox0beta / math.expm1(xox0beta)
    except OverflowError:
        # If xox0beta is very large, then the bterm is zero
        bterm = 0.0
    return x - (1.0 - math.exp(-x)) * (3.0 + alpha + beta * bterm)  


class modified_blackbody(object):
    """A class representing a modified greybody

    The form for the modified blackbody is
    
    .. math::

      f_{\\nu} \\propto \\left(1 - \\exp\\left[ - \\left(\\nu / 
      \\nu_0\\right)^{\\beta} B_{\\nu}\\left( \\nu ; T \\right)

    where :math:`B_{\\nu}` is the Planck blackbody function in frequency
    units.  Class instances are static.
    """

    def __init__(self, T, beta, lambda0, alpha, fnorm, wavenorm=500.0,
                 noalpha=False, opthin=False):
        """Initializer

        Parameters:
        -----------
        T : float
          Temperature/(1+z) in K
        
        beta : float
          Extinction slope

        lambda0 : float
          Wavelength where emission becomes optically thick * (1+z), in 
          microns

        alpha : float
          Blue side power law slope

        fnorm : float
          Normalization flux, in mJy

        wavenorm : float
          Wavelength of normalization flux, in microns (def: 500)

        noalpha : bool
          Do not use blue side power law

        opthin : bool
          Assume emission is optically thin
        """

        self._T = float(T)
        self._beta = float(beta)
        self._lambda0 = float(lambda0)
        self._hasalpha = not bool(noalpha)
        self._alpha = float(alpha)
        self._fnorm = float(fnorm)
        self._wavenorm = float(wavenorm)
        if bool(opthin):
            self._opthin = True
        else:
            self._opthin = False
            
        if self._hasalpha and alpha <= 0.0:
            errmsg = "alpha must be positive.  You gave: %.5g" % self._alpha
            raise ValueError(errmsg)
        if self._beta <= 0.0:
            errmsg = "beta must be positive.  You gave: %.5g" % self._beta
            raise ValueError(errmsg)

        # Some constants
        c = 299792458e6 #in microns
        h = 6.6260693e-34 #J/s
        k = 1.3806505e-23 #J/K
        hcokt = h * c / (k * self._T)

        # Convert wavelengths to x = h nu / k T
        self._x0 = hcokt / lambda0
        self._xnorm = hcokt / self._wavenorm

        # Two cases -- optically thin and not.
        #  Each has two sub-cases -- with power law merge and without
        if self._opthin:
            if not self._hasalpha:
                # No merge to power law, easy
                self._normfac = self._fnorm * math.expm1(self._xnorm) / \
                    self._xnorm**(3.0 + beta)
            else:
                # First, figure out the x (frequency) where the join
                # happens.  At frequencies above this (x > xmarge)
                # are on the blue, alpha power law side
                # The equation we are trying to find the root for is:
                #  x - (1-exp(-x))*(3+alpha+beta)
                # Amazingly, this has a special function solution
                #   A = (3+alpha+beta)
                #   xmerge = A + LambertW[ -A Exp[-A] ]
                # This has a positive solution for all A > 1 -- and since
                # we require alpha and beta > 0, this is always the case
                a = 3.0 + self._alpha + self._beta
                self._xmerge = a + lambertw(-a * math.exp(-a)).real

                # Get merge constant -- note this is -before- flux normalization
                # to allow for the case where wavenorm is on the power law part
                self._kappa = self._xmerge**(3.0 + self._alpha + self._beta) / \
                    math.expm1(self._xmerge)

                # Compute normalization constant
                if self._xnorm > self._xmerge :
                    self._normfac = self._fnorm * self._xnorm**self._alpha / \
                        self._kappa
                else :
                    self._normfac = self._fnorm * math.expm1(self._xnorm) / \
                        self._xnorm**(3.0 + self._beta)
                
        else:
            #Optically thick case
            if not self._hasalpha:
                self._normfac = - self._fnorm * math.expm1(self._xnorm) / \
                    (math.expm1(-(self._xnorm / self._x0)**self._beta) * \
                         self._xnorm**3)
            else :
                # This is harder, and does not have a special function
                # solution.  Hence, we have to do this numerically.
                # The equation we need to find the root for is given by 
                # alpha_merge_eqn.  

                # First, we bracket.  For positive alpha, beta
                # we expect this to be negative for small a and positive
                # for large a.  We try to step out until we achieve that
                maxiters = 100
                a = 0.1
                aval = alpha_merge_eqn(a, self._alpha, self._beta, self._x0)
                iter = 0
                while aval >= 0.0:
                    a /= 2.0
                    aval = alpha_merge_eqn(a, self._alpha, self._beta, self._x0)
                    if iter > maxiters:
                        errmsg = "Couldn't bracket low alpha merge point for "\
                            "T: %f beta: %f lambda0: %f alpha %f, "\
                            "last a: %f value: %f"
                        errmsg %= (self._T, self._beta, self._lambda0, 
                                   self._alpha, a, aval)
                        raise ValueError(errmsg)
                    iter += 1

                b = 15.0
                bval = alpha_merge_eqn(b, self._alpha, self._beta, self._x0)
                iter = 0
                while bval <= 0.0:
                    b *= 2.0
                    bval = alpha_merge_eqn(b, self._alpha, self._beta, self._x0)
                    if iter > maxiters:
                        errmsg = "Couldn't bracket high alpha merge point for "\
                            "T: %f beta: %f lambda0: %f alpha %f, "\
                            "last b: %f value: %f"
                        errmsg %= (self._T, self._beta, self._lambda0, 
                                   self._alpha, b, bval)
                        raise ValueError(errmsg)
                    iter += 1
                    
                # Now find root
                args = (self._alpha, self._beta, self._x0)
                self._xmerge = scipy.optimize.brentq(alpha_merge_eqn, a, b,
                                                     args=args, disp=True)

                #Merge constant
                # Note this will overflow and crash for large xmerge, alpha
                self._kappa = - self._xmerge**(3 + self._alpha) * \
                    math.expm1(-(self._xmerge / self._x0)**self._beta) / \
                    math.expm1(self._xmerge)

                #Normalization factor
                if self._xnorm > self._xmerge :
                    self._normfac = self._fnorm * self._xnorm**self._alpha / \
                        self._kappa
                else :
                    self._normfac = - self._fnorm * math.expm1(self._xnorm) / \
                        (self._xnorm**3 * \
                             math.expm1(-(self._xnorm / self._x0)**self._beta))

    @property
    def T(self):
        return self._T

    @property
    def beta(self):
        return self._beta
    
    @property
    def lambda0(self):
        return self._lambda0

    @property
    def alpha(self):
        return self._alpha
    
    @property
    def fnorm(self):
        return self._fnorm

    @property
    def wavenorm(self):
        return self._wavenorm

    @property
    def has_alpha(self):
        return self._hasalpha

    @property
    def optically_thin(self):
        return self._opthin

    def __repr__(self):
        retstr = "T: %.4g beta: %.4g lambda0: %.4g alpha: %.4g fnorm: %.4g"
        retstr += " noalpha: %s opthin: %s"
        return retstr % (self._T, self._beta, self._lambda0, self._alpha,
                         self._fnorm, not self._hasalpha, self._opthin)

    def f_nu(self, freq):
        """Evaluate modifed blackbody at specified frequencies.

        Parameters
        ----------
        freq : array_like
          Input frequencies, in GHz

        Returns
        -------
        fnu : ndarray, or float if input scalar
          The flux density in mJy
        """

        # Convert to some form of numarray
        if not isiterable(freq):
            frequency = numpy.asarray([freq], dtype=numpy.float)
        else:
            frequency = numpy.asanyarray(freq, dtype=numpy.float)

        # Some constants
        h = 6.6260693e-34 #J/s
        k = 1.3806505e-23 #J/K
        hokt = h / (k * self._T)

        # Convert wavelengths to x = h nu / k T
        x = hokt * 1e9 * frequency  #1e9 to convert to Hz from GHz

        # Two cases -- optically thin and not.
        #  Each has two sub-cases -- with power law merge and without
        if self._opthin:
            if not self._hasalpha:
                retval = self._normfac * x**(3.0 + self._beta) / numpy.expm1(x)
            else:
                retval = numpy.zeros_like(frequency)
                ispower = x > self._xmerge
                retval[ispower] = self._kappa * x[ispower]**(-self._alpha)
                retval[~ispower] = x[~ispower]**(3.0 + self._beta) / \
                    numpy.expm1(x[~ispower])
                retval *= self._normfac
        else:
            if not self._hasalpha:
                retval = - self._normfac * \
                    numpy.expm1(-(x / self._x0)**self._beta) * x**3 / \
                    numpy.expm1(x)
            else :
                retval = numpy.zeros_like(frequency)
                ispower = x > self._xmerge
                retval[ispower] = self._kappa * x[ispower]**(-self._alpha)
                retval[~ispower] = \
                    - numpy.expm1( - (x[~ispower]/self._x0)**self._beta) * \
                    x[~ispower]**3/numpy.expm1(x[~ispower])
                retval *= self._normfac
        return retval

    def __call__(self, wave):
        """Evaluate modified blackbody at specified wavelengths

        Parameters
        ----------
        wave : array_like
          Input wavelengths, in microns

        Returns
        -------
        fnu : ndarray, or float if input scalar
          The flux density in mJy
        """
        
        c = 299792458e-3 #The microns to GHz conversion
        wviter = isiterable(wave)
        if wviter:
            wave = numpy.asanyarray(wave, dtype=numpy.float)
            return self.f_nu(c / wave)
        else:
            return self.f_nu(c / float(wave))

    def freq_integrate(self, minwave, maxwave):
        """Integrate f_nu over specified wavelength range

        Parameters:
        -----------
        minwave : float
          Minimum wavlength, in microns

        maxwave : float
          Maximum wavelength, in microns

        Returns
        -------
        fint : float
          The integral in erg/s/cm^2
        """
        
        from scipy.integrate import quad

        minwave = float(minwave)
        maxwave = float(maxwave)

        if minwave <= 0.0:
            raise ValueError("Minimum wavelength must be > 0.0")
        if minwave > maxwave:
            minwave, maxwave = maxwave, minwave
        
        # Tricky thing -- we are integrating over frequency (in GHz),
        # not wavelength
        c = 299792458e-3
        minfreq = c / maxwave
        maxfreq = c / minwave

        fint = quad(self.f_nu, minfreq, maxfreq)[0]

        # Integral comes back in mJy-GHz, convert to erg/s/cm^2
        return 1e-17 * fint
        
