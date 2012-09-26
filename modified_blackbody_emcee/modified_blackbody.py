import math
import numpy
import scipy.optimize

from utility import isiterable

"""Modified blackbody SED"""

def alpha_merge_eqn_opthin(x, alpha, beta):
    """Equation we need the root for to merge power law to modified
    blackbody, optically thin version"""
    return x - (1.0 - math.exp(-x)) * (3.0 + alpha + beta)

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
        T : (float)
          Temperature/(1+z) in K
        
        beta : (float)
          Extinction slope

        lambda0 : (float)
          Wavelength where emission becomes optically thick * (1+z), in 
          microns

        alpha : (float)
          Blue side power law slope

        fnorm : (float)
          Normalization flux, in mJy

        wavenorm : (float)
          Wavelength of normalization flux, in microns (def: 500)

        noalpha : (bool)
          Do not use blue side power law

        opthin : (bool)
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
            
        #Set up stuff like normalization, merge
        # Some constants
        c = 299792458e6 #in microns
        h = 6.6260693e-34 #J/s
        k = 1.3806505e-23 #J/K
        hcokt = h * c / (k * self._T)

        # Convert wavelengths to x = h nu / k T
        self._x0 = hcokt / lambda0
        self._xnorm = hcokt / wavenorm

        # Two cases -- optically thin and not.
        #  Each has two sub-cases -- with power law merge and without
        if self._opthin:
            if not self._hasalpha:
                # No merge to power law, easy
                self._normfac = self._fnorm * math.expm1(self._xnorm) / \
                    self._xnorm**(3.0 + beta)
            else:
                # First, figure out the x (frequency) where the join
                # happens.  Ass frequencies above this (x > xmarge)
                # are on the blue, alpha power law side
                # This has to be done numerically
                a = 0.01
                b = 30.0
                aval = alpha_merge_eqn_opthin(a, self._alpha, self._beta)
                bval = alpha_merge_eqn_opthin(b, self._alpha, self._beta)

                if (aval * bval > 0) :  #Should have opposite signs!
                    errmsg="Couldn't find alpha merge point for T: "\
                        "%f beta: %f alpha %f, f(%f): %f f(%f): %f"
                    errmsg %= (self._T,self._beta,self._alpha,a,aval,b,bval)
                    raise ValueError(errmsg)

                args = (self._alpha, self._beta)
                self._xmerge = scipy.optimize.brentq(alpha_merge_eqn_opthin, 
                                                     a, b, args=args,
                                                     disp=True)

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
                    (math.expm1(-(self._xnorm / self._x0)**beta) * \
                         self._xnorm**3)
            else :
                a = 0.01
                b = 30.0
                aval = alpha_merge_eqn(a, self._alpha, self._beta, self._x0)
                bval = alpha_merge_eqn(b, self._alpha, self._beta, self._x0)
                if (aval * bval > 0) :  #Should have opposite signs!
                    errmsg="Couldn't find alpha merge point for T: "\
                        "%f beta: %f lambda0: %f alpha %f, f(%f): %f f(%f): %f"
                    errmsg %= (self._T, self._beta, self._lambda0, 
                               self._alpha, a, aval, b, bval)
                    raise ValueError(errmsg)
                
                args = (self._alpha, self._beta, self._x0)
                self._xmerge = scipy.optimize.brentq(alpha_merge_eqn, a, b,
                                                     args=args, disp=True)

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

        wviter = isiterable(wave)
        if wviter:
            wave = numpy.asarray(wave)

        # Some constants
        c = 299792458e6 #in microns
        h = 6.6260693e-34 #J/s
        k = 1.3806505e-23 #J/K
        hcokt = h * c / (k * self._T)

        # Convert wavelengths to x = h nu / k T
        x = hcokt / numpy.array(wave)
        
        # Two cases -- optically thin and not.
        #  Each has two sub-cases -- with power law merge and without
        if self._opthin:
            if not self._hasalpha:
                retval = self._normfac * x**(3.0 + self._beta) / numpy.expm1(x)
            else:
                retval = numpy.zeros_like(wave)
                if wviter:
                    ispower = x > self._xmerge
                    retval[ispower] = self._kappa * x[ispower]**(-self._alpha)
                    retval[~ispower] = x[~ispower]**(3.0 + self._beta) / \
                        numpy.expm1(x[~ispower])
                    retval *= self._normfac
                else:
                    if x > self._xmerge:
                        retval = self._normfac * self._kappa * x**(-self._alpha)
                    else:
                        retval = self._normfac * x**(3.0 + self._beta) / \
                            numpy.expm1(x)
        else:
            if not self._hasalpha:
                retval = - self._normfac * \
                    numpy.expm1(-(x / self._x0)**self._beta) * x**3 / \
                    numpy.expm1(x)
            else :
                retval = numpy.zeros_like(wave)
                if wviter:
                    ispower = x > self._xmerge
                    retval[ispower] = self._kappa * x[ispower]**(-self._alpha)
                    retval[~ispower] = \
                        - numpy.expm1( - (x[~ispower]/self._x0)**self._beta) * \
                        x[~ispower]**3/numpy.expm1(x[~ispower])
                    retval *= self._normfac
                else:
                    if x > self._xmerge:
                        retval = self._normfac * self._kappa * \
                            x**(-self._alpha)
                    else:
                        retval = -self._normfac * x**3 * \
                            numpy.expm1( - (x / self._x0)**self._beta) /\
                            numpy.expm1(x)
        return retval
