import math
import numpy as np

"""Astronomical instrument response modelling"""

__all__ = ["response", "response_set"]

def response_bb(freq, temperature):
    """Evaluates blackbody fnu at specified frequency and temperature

    Parameters
    ----------
    freq : float or ndarray
      Frequencies to evaluate at in GHz

    temperature : float
      Temperature in Kelvin

    Returns
    -------
    fnu : float or ndarray
      Blackbody function, ignoring normalization.
    """

    # Some constants -- eventually, replace these with
    # astropy.constants, but that is in development, so hardwire for now
    h = 6.6260693e-34 #J/s
    k = 1.3806505e-23 #J/K
    hokt = 1e9 * h / (k * float(temperature)) #The 1e9 is GHz -> Hz
    
    return freq**3 / numpy.expm1(hokt * freq)

class response(object):
    """ A class representing the response of an instrument to an observation.

    Handles instrument responses and pipeline normalization conventions.
    """

    def __init__(self, inputfile, xtype='wave', xunits='microns',
                 normtype = 'power', xnorm=250.0, normparam=-1.0):
        """ Read in the input file and set up the normalization

        Parameters
        ----------
        inputfile : string
          Name of input file.  Must be a text file.

        xtype : string
          Type of input x variable.  lambda for wavelengths, freq for frequency.

        xunits : string
          Units of input x variable.  microns, angstroms, or meters for 
          wavelength, hz, mhz, ghz, thz for frequency.
        
        normtype : string
          Type of normalization.  power for power-law, bb for blackbody.

        xnorm : float
           X value of normalization, same units as input x value.

        normparam : float
          Normalizatin parameter -- power low exponent for power-law,
          temperature for blackbody
        """

        import astropy.io.ascii
        
        if not isinstance(inputfile, basestring):
            raise TypeError("filename must be string-like")
        
        data = astropy.io.ascii.read(inputfile, comment='^#')
        if len(data) == 0 :
            raise IOError("No data read from %s" % inputfile)

        xvals = numpy.asarray([dat[0] for dat in data])
        self._resp = numpy.asarray([dat[1] for dat in data])

        # We don't allow negative responses or xvals
        if xvals.min() <= 0:
            raise ValueError("Non-positive x value encountered")
        if self._resp.min() <= 0:
            raise ValueError("Non-positive response encountered")
        if xnorm <= 0:
            raise ValueError("Non-positive xnorm")

        #Build up wavelength/normwave in microns, freq in GHz
        xtyp = xtype.lower()
        xun = xunits.lower()
        if xtyp == 'wave':
            if xun == 'angstroms':
                self._wave = 1e-4 * xvals
                self._normwave = 1e-4 * xnorm
            elif xun == 'microns':
                self._wave = xvals
                self._normwave = xnorm
            elif xun == 'meters':
                self._wave = 1e6 * xvals
                self._normwave = 1e6 * xnorm
            else:
                raise ValueError("Unrecognized wavelength type %s" % xtype)
            self._freq = 299792458e-3 / self._wave #In GHz
            self._normfreq = 299792458e-3 / self._normwave
        elif xtyp == 'freq':
            # Change to GHz
            if xun == 'hz':
                self._freq = 1e-9 * xvals
                self._normfreq = 1e-9 * xnorm
            elif xun == 'mhz':
                self._freq = 1e-3 * xvals
                self._normfreq = 1e-3 * xnorm
            elif xun == 'ghz':
                self._freq = xvals
                self._normfreq = xnorm
            elif xun == 'thz':
                self._freq = 1e3 * xvals
                self._normfreq = 1e3 * xnorm
            self._wave = 299792458e-3 / self._freq #Microns
            self._normwave = 299792458e-3 / self._normfreq

        # Sort into ascending wavelength order for simplicity
        idx = self._wave.argsort()
        self._wave = self._wave[idx]
        self._freq = self._freq[idx]
        self._resp = self._resp[idx]
        self._nresp = len(self._resp)

        # This is a convenience array for integrations, consisting of
        # the response times the delta freq factor to sum over
        # Note that integrations will be done in frequency always.
        # Since freq is in reverse order, these will be negative
        self._dnu = self._freq[1:self._nresp] - self._freq[0:self._nresp-2]
        self._sedmult = np.zeros(self._nresp)
        self._sedmult[0:self._nresp-1] = 0.5 * self._dnu
        self._sedmult[self._nresp-1] = 0.5 * self._dnu[self._nresp-1]
        self._sedmult[1:self._nresp-1] += 0.5 * self._dnu[0:self._nresp-2]
        self._sedmult *= self._resp

        #And, now the normalization condition

        ntype = normtype.lower()
        if ntype == "power":
            sed = (self._freq / self._normfreq)**normparam
        elif ntype == "bb":
            # Rather more complicated
            nrmpar = float(normparam)
            if nrmpar <= 0.0:
                errmsg = "Invalid (non-positive) blackbody temperature %f"
                raise ValueError(errmsg % nrmpar)
            # Blackbody curve, normalized at normfreq
            sed = response_bb(self._freq, nrmpar) /\
                response_bb(self._normfreq, nrmpar)
        else:
            raise ValueError("Unknown normalization type %s" % normtype)

        # Note this will be negative due to the frequency ordering,
        # but that's okay because our normal integrals will be negative
        # as well so when we divide them it will all work out.
        self._normfac = (sed * self._stepmult).sum()

    @property
    def wavelength(self):
        """ Wavelength of response function in microns"""
        return self._wave

    @property
    def frequency(self):
        """ Frequency of response function in GHz"""
        return self._freq

    @property
    def response(self):
        """ Response function at wavelengths specified by wave"""
        return self._resp

    @property
    def normfac(self):
        """ Normalization value"""
        # Multiply by -1 to avoid confusion
        return -1.0 * self._normfac


    def __call__(self, fnufunc, freq=False):
        """ Gets the instrumental response for a SED.

        Parameters
        ----------
        fnufunc : function
          A function which takes wavelengths in microns and produces
          f_nu in the target units (usually mJy or Jy).

        freq : bool
          If set, function takes frequency in GHz instead.

        Returns
        -------
        resp : float
          Instrument response, including pipeline normalization convention.
        """
        
        if freq is False:
            # Wavelength units
            return (fnufunc(self._wave) * self._stepmult).sum() / self._normfac
        else:
            # Frequency
            return (fnufunc(self._freq) * self._stepmult).sum() / self._normfac


class response_set(object):
    """ A set of instrument responses"""
    
    def __init__(self, inputfile):
        """ Initialize response set.

        Parameters
        ----------
        inputfile : string
          Name of input file
        """

        import astropy.io.ascii
        
        if not isinstance(inputfile, basestring):
            raise TypeError("filename must be string-like")
        
        data = astropy.io.ascii.read(inputfile, comment='^#')
        if len(data) == 0 :
            raise IOError("No data read from %s" % inputfile)

        self._reponses = {}
        for dat in data:
            name = dat[0]
            self._responses[name] = response(dat[1], dat[2].lower(),
                                             dat[3].lower(), dat[4].lower(),
                                             float(dat[5]), float(dat[6]))

    def __getitem__(self, name):
        """ Get response with a specified name

        Parameters
        ----------
        name : string
          Name identifying response function.
          
        Returns
        -------
        resp : response
          Class object giving response function.
        """
        return self._responses[name]
