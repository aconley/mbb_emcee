import math
import numpy

"""Astronomical instrument response modeling"""

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
                 normtype = 'power', xnorm=250.0, normparam=-1.0,
                 filtdir=None):
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
          Type of normalization.  power for power-law, bb for blackbody, flat
          for flat fnu, delta for delta function, none for no normalization, etc..

        xnorm : float
           X value of normalization, same units as input x value.

        normparam : float
          Normalizatin parameter -- power low exponent for power-law,
          temperature for blackbody
    
        Notes
        -----
        If normtype is delta, then the filter function is assumed to be a delta
        function at xnorm, and no filter file is read.
        """

        import astropy.io.ascii
        import os.path

        # Make sure all args have right case
        ntype = normtype.lower()
        xtyp = xtype.lower()
        xun = xunits.lower()

        #Delta functions are a special case
        if ntype == "delta":
            # Quick, special return
            if xnorm <= 0:
                raise ValueError("Non-positive xnorm")
            if xtyp == 'wave':
                if xun == 'angstroms':
                    self._normwave = 1e-4 * xnorm
                elif xun == 'microns':
                    self._normwave = xnorm
                elif xun == 'meters':
                    self._normwave = 1e6 * xnorm
                else:
                    raise ValueError("Unrecognized wavelength type %s" % xtype)
                self._normfreq = 299792458e-3 / self._normwave
            elif xtyp == 'freq':
                # Change to GHz
                if xun == 'hz':
                    self._normfreq = 1e-9 * xnorm
                elif xun == 'mhz':
                    self._normfreq = 1e-3 * xnorm
                elif xun == 'ghz':
                    self._normfreq = xnorm
                elif xun == 'thz':
                    self._normfreq = 1e3 * xnorm
                self._normwave = 299792458e-3 / self._normfreq
            self._isdelta = True
            self._resp = numpy.array([1.0])
            self._nresp = 1
            self._normtype = "delta"
            self._normparam = None
            self._normfac = 1.0

            return

        #Read in data
        if not isinstance(inputfile, basestring):
            raise TypeError("filename must be string-like")
        
        if filtdir is None:
            data = astropy.io.ascii.read(inputfile, comment='^#')
            if len(data) == 0 :
                raise IOError("No data read from %s" % inputfile)
        else:
            infile = os.path.join(filtdir, inputfile)
            data = astropy.io.ascii.read(infile, comment='^#')
            if len(data) == 0 :
                raise IOError("No data read from %s" % infile)

        self._isdelta = False
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
        self._dnu = self._freq[1:self._nresp] - self._freq[0:self._nresp-1]
        self._sedmult = numpy.empty(self._nresp)
        self._sedmult[0:self._nresp-1] = 0.5 * self._dnu
        self._sedmult[self._nresp-1] = 0.5 * self._dnu[self._nresp-2]
        self._sedmult[1:self._nresp-1] += 0.5 * self._dnu[0:self._nresp-2]
        self._sedmult *= self._resp

        #And, now the normalization condition
        self._normtype = ntype
        if ntype == "none":
            self._normparam = None
            self._normfac = -1.0
        else:
            if ntype == "power":
                self._normparam = float(normparam)
                sed = (self._freq / self._normfreq)**self._normparam
            elif ntype == "flat":
                sed = numpy.ones(self._nresp)
                self._normparam = None
            elif ntype == "bb":
                # Rather more complicated
                self._normparam = float(normparam)
                if self._normparam <= 0.0:
                    errmsg = "Invalid (non-positive) blackbody temperature %f"
                    raise ValueError(errmsg % self._normparam)
                # Blackbody curve, normalized at normfreq
                sed = response_bb(self._freq, self._normparam) /\
                    response_bb(self._normfreq, self._normparam)
            else:
                raise ValueError("Unknown normalization type %s" % normtype)

            # Note this will be negative due to the frequency ordering,
            # but that's okay because our normal integrals will be negative
            # as well so when we divide them it will all work out.
            self._normfac = 1.0 / (sed * self._sedmult).sum()

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
        
        if self._isdelta:
            if freq:
                return fnufunc(self._normfreq)
            else:
                return fnufunc(self._normwave)
        else:
            if freq:
                return (fnufunc(self._freq) * self._sedmult).sum() *\
                    self._normfac
            else:
                return (fnufunc(self._wave) * self._sedmult).sum() *\
                    self._normfac

class response_set(object):
    """ A set of instrument responses"""
    
    def __init__(self, inputfile, filtdir=None):
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

        self._responses = {}
        for dat in data:
            name = dat[0]
            self._responses[name] = response(dat[1], dat[2].lower(),
                                             dat[3].lower(), dat[4].lower(),
                                             float(dat[5]), float(dat[6]),
                                             filtdir=filtdir)

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

    def setitem(self, val):
        if not isinstance(val, response):
            raise ValueError("Val not of type mbb_emcee.response")
        return self._responses.setitem(val)

    def keys(self):
        return self._responses.keys()

    def viewkeys(self):
        return self._responses.viewkeys()

    def has_key(self, val):
        return self._responses.has_key(val)

    def iterkeys(self):
        return self._responses.iterkeys()

    def items(self):
        return self._responses.items()

    def iteritems(self):
        return self._responses.iteritems()

    def viewitems(self):
        return self._responses.viewitems()

    def viewvalues(self):
        return self._responses.viewvalues()

    def itervalues(self):
        return self._responses.itervalues()

    def contains(self, val):
        return self._responses.contains(val)

    def delitem(self, val):
        return self._responses.delitem(val)

    def clear(self):
        return self._respones.clear()
