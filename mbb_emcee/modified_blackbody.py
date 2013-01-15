import math
import numpy
import scipy.optimize
from scipy.special import lambertw

from utility import isiterable

"""Modified blackbody SED"""

__all__ = ["modified_blackbody"]

def alpha_merge_eqn(x, alpha, beta, x0, opthin=False):
    """Equation we need the root for to merge power law to modified
    blackbody

    Parameters
    ----------
    x : float
      h nu / k T to evaluate at

    alpha : float
      blue side power law index

    beta : float
      Dust attenuation power law index

    x0 : float
      h nu_0 / k T

    opthin : bool
      Assume optically thin case
    """

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
        if bool(noalpha):
            self._hasalpha = False
            self._alpha = None
        else:
            self._hasalpha = True
            self._alpha = float(alpha)

        self._fnorm = float(fnorm)
        self._wavenorm = float(wavenorm)

        if bool(opthin):
            self._opthin = True
            self._lambda0 = None
        else:
            self._opthin = False
            self._lambda0 = float(lambda0)
            
        if self._hasalpha and alpha <= 0.0:
            errmsg = "alpha must be positive.  You gave: %.5g" % self._alpha
            raise ValueError(errmsg)
        if self._beta < 0.0:
            errmsg = "beta must be non-negative.  You gave: %.5g" % self._beta
            raise ValueError(errmsg)

        # Some constants
        c = 299792458e6 #in microns
        h = 6.6260693e-34 #J/s
        k = 1.3806505e-23 #J/K
        self._hcokt = h * c / (k * self._T)

        # Convert wavelengths to x = h nu / k T
        if not self._opthin:
            self._x0 = self._hcokt / lambda0
        self._xnorm = self._hcokt / self._wavenorm

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
        """ Get temperature / (1+z) in K"""
        return self._T

    @property
    def beta(self):
        """ Get Beta"""
        return self._beta
    
    @property
    def lambda0(self):
        """ Get lambda_0 (1+z) in microns"""
        if self._opthin: return None
        return self._lambda0

    @property
    def alpha(self):
        """ Get alpha"""
        if self._noalpha: return None
        return self._alpha
    
    @property
    def fnorm(self):
        """ Get normalization flux at wavenorm in mJy"""
        return self._fnorm

    @property
    def wavenorm(self):
        """ Get normalization flux wavelength in microns"""
        return self._wavenorm

    @property
    def has_alpha(self):
        """ Does this modified_blackbody use a blue side power law?"""
        return self._hasalpha

    @property
    def optically_thin(self):
        """ Does this modified_blackbody assume it is optically thin?"""
        return self._opthin

    @property
    def wavemerge(self):
        """Get the merge wavelength in microns"""
        if not self._hasalpha:
            return None
        else:
            return self._hcokt / self._xmerge

    def __repr__(self):
        if self._hasalpha:
            if self._opthin:
                retstr = "modified_blackbody(%.2g, %.2g, None, %.2g, %.2g," + \
                    " opthin=True)"
                return retstr % (self._T, self._beta, self._alpha,
                                 self._fnorm)
            else:
                retstr = "modified_blackbody(%.2g, %.2g, %.2g, %.2g, %.2g," + \
                    " opthin=True)"
                return retstr % (self._T, self._beta, self.lambda0,
                                 self.alpha, self._fnorm)
        else:
            if self._opthin:
                retstr = "modified_blackbody(%.2g, %.2g, None, None, %.2g," + \
                    " noalpha=True, opthin=True)"
                return retstr % (self._T, self._beta, self._fnorm)
            else:
                retstr = "modified_blackbody(%.2g, %.2g, %.2g, None, %.2g," + \
                    " noalpha=True)"
                return retstr % (self._T, self._beta, self.lambda0,
                                 self._fnorm)


    def __str__(self):
        if self._hasalpha:
            if self._opthin:
                retstr = "modified_blackbody(T: %.2g beta: %.2g " + \
                    "alpha: %.2g fnorm: %.2g)"
                return retstr % (self._T, self._beta, self._alpha,
                                 self._fnorm)
            else:
                retstr = "modified_blackbody(T: %.2g beta: %.2g " + \
                    "lambda0: %.2g alpha: %.2g fnorm: %.2g)"
                return retstr % (self._T, self._beta, self.lambda0,
                                 self._alpha, self._fnorm)
        else:
            if self._opthin:
                retstr = "modified_blackbody(T: %.2g beta: %.2g fnorm: %.2g)"
                return retstr % (self._T, self._beta, self._fnorm)
            else:
                retstr = "modified_blackbody(T: %.2g beta: %.2g " + \
                    "lambda0: %.2g fnorm: %.2g)"
                return retstr % (self._T, self._beta, self.lambda0,
                                 self._fnorm)  

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
        """ Evaluate modified blackbody at specified wavelengths

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

    def _snudev(self, x):
        """ Evaluates derivative (modulo normalization) of S_nu at x

        x = h nu / k T (scalar)

        Ignores alpha side, since that should be rising -- so the peak
        should lie at lower frequency than the merge to the alpha law"""
        
        if self._opthin:
            efac = math.expm1(x)
            return x**(2.0 + self._beta) * (3.0 + self._beta) / efac - \
                math.exp(x) * x**(3.0 + self._beta) / efac**2
        else:
            efac = math.expm1(x)
            xx0 = x / self._x0
            try:
                xx0b = xx0**self._beta
                ebfac = - math.expm1(-xx0b)
                return 3 * x**2 * ebfac / efac -\
                    math.exp(x) * x**3 * ebfac / efac**2 +\
                    self._beta * x**3 * math.exp(-xx0b) * xx0b / \
                    (x * efac)
            except OverflowError:
                # (x/x0)**beta is too large, which simplifies the expression
                return 3 * x**2 / efac - math.exp(x) * x**3 / efac**2

    def max_wave(self):
        """ Get the wavelength of maximum emission in f_nu units.

        Returns
        -------
        wave : float
         The wavelength of the maximum in microns
        """
        
        # Note that the alpha portion is ignored, since we
        # require alpha to be positive.  That means that
        # the power law part should be rising where it joins
        # the modified blackbody part, and therefore it should
        # not affect anything

        from scipy.optimize import brentq

        # Start with an expression for the maximum of a normal
        # blackbody.  We work in x = h nu / k T 
        c = 299792458e6 #in microns
        h = 6.6260693e-34 #J/s
        k = 1.3806505e-23 #J/K
        xmax_bb = 2.82144
        numax_bb = xmax_bb * k * self._T / h
        if (self._opthin and self._beta == 0):
            # This is just a blackbody, so easy cakes
            return c / numax_bb

        # Now, bracket the root in the derivative.
        # At low x (low frequency) the derivative should be positive
        a = xmax_bb / 2.0
        aval = self._snudev(a)
        maxiters = 20
        iter = 0
        while aval <= 0.0:
            if iter > maxiters:
                errmsg = "Couldn't bracket maximum from low frequency side"
                raise Exception(errmsg)
            a /= 2.0
            aval = self._snudev(a)
            iter += 1

        # And the derivative should be negative at high frequencies
        b = xmax_bb * 2.0
        bval = self._snudev(b)
        iter = 0
        while bval >= 0.0:
            if iter > maxiters:
                errmsg = "Couldn't bracket maximum from high frequency side"
                raise Exception(errmsg)
            b *= 2.0
            bval = self._snudev(b)
            iter += 1
        
        # Now find the actual root
        xmax = brentq(self._snudev, a, b, disp=True)

        # Convert to more conventional units
        numax = xmax * k * self._T / h
        return c / numax

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
        
