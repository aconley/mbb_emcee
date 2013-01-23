import math
import numpy
import os.path

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

    def __init__(self, name, inputfile, xtype='wave', xunits='microns',
                 senstype='energy', normtype='power', xnorm=250.0, 
                 normparam=-1.0, dir=None):
        """ Read in the input file and set up the normalization

        Parameters
        ----------
        name : string
          Brief name of response function

        inputfile : string
          Name of input file.  Must be a text file.  If this has the special
          values delta_?, boxcar_?, or gauss_?, then rather than reading from a file
          the passband is created internally.

        xtype : string
          Type of input x variable.  lambda for wavelengths, freq for frequency.

        xunits : string
          Units of input x variable.  microns, angstroms, or meters for 
          wavelength, hz, mhz, ghz, thz for frequency.
        
        normtype : string
          Type of normalization.  power for power-law, bb for blackbody, flat
          for flat fnu, none for no normalization, etc.

        senstype : string
          Type of sensitivity -- energy or counts.

        xnorm : float
           X value of normalization, same units as input x value.

        normparam : float
          Normalizatin parameter -- power low exponent for power-law,
          temperature for blackbody
    
        dir : string
          Directory to look for inputfile in.  Defaults to current directory.

        Notes
        -----
        If inputfile is delta_val, then the filter function is assumed to be a delta
        function at value.  normtype, etc. is ignored.  For example, if xunit
        is microns, then delta_880 sets up a delta function at 880um.  If xunit
        were GHz, then it would be a delta function at 880GHz.

        If inputfile is boxcar_val1_val2, then the filter function is a box with 11 points
        centered on val1 and with a width of val2 (in the units of xunits)

        If inputfile is gauss_val1_val2, then the filter function is a Gaussian centered
        on val1 with a FWHM of val2.  It is sampled every FWHM/7 steps
        out to 3*FWHM in each direction.

        So don't name your input files delta_?, gauss_?, or box_?
        """

        import astropy.io.ascii
        import scipy.integrate

        self._name = name

        # Make sure all args have right case
        ntyp = normtype.lower()
        xtyp = xtype.lower()
        xun = xunits.lower()
        styp = senstype.lower()

        # Read in or set-up the data
        # If there is a _, we may have a special case.
        # delta function allows quick return
        #Read in data or setup special case
        if not isinstance(inputfile, basestring):
            raise TypeError("filename must be string-like")
        self._isdelta = False
        if inputfile.find('_'):
            #First, check if it's a delta.  If so, do the
            # quick return setup for that
            spl = inputfile.split('_')
            bs = spl[0].lower()
            if bs == "delta":
                if len(spl) != 2:
                    raise ValueError("delta function has too many params")
                val = float(spl[1])
                self._setup_delta(val, xtyp, xun)
                return
            elif bs == "box":
                if len(spl) != 3:
                    raise ValueError("box car needs 2 params")
                cent = float(spl[1])
                width = float(spl[2])
                npoints = 11
                xvals = numpy.linspace(cent - 0.5 * width,
                                       cent + 0.5 * width,
                                       npoints)
                self._resp = numpy.ones(npoints)
            elif bs == "gauss":
                if len(spl) != 3:
                    raise ValueError("gaussian needs 2 params")
                cent = float(spl[1])
                fwhm = float(spl[2])
                minval = cent - 3.0*fwhm
                maxval = cent + 3.0*fwhm
                npoints = 43 # (FWHM/7)
                sig = fwhm / math.sqrt(8 * math.log(2))
                xvals = numpy.linspace(cent - 3.0*fwhm, 
                                       cent + 3.0*fwhm,
                                       npoints)
                self._resp = numpy.exp(-0.5 * ((xvals - cent) / sig)**2)
            else:
                #A real file -- read it
                if dir is None:
                    data = astropy.io.ascii.read(inputfile, comment='^#')
                    if len(data) == 0 :
                        raise IOError("No data read from %s" % inputfile)
                else:
                    infile = os.path.join(dir, inputfile)
                    data = astropy.io.ascii.read(infile, comment='^#')
                    if len(data) == 0 :
                        raise IOError("No data read from %s" % infile)

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

        # Set up sensitivity type
        if styp == "energy":
            self._sens_energy = True
        elif styp == "counts":
            self._sens_energy = False
        else:
            raise ValueError("Unknown sensitivity type %s" % senstype)

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

        # If the sensitivity type is counts, then we need to divide sedmult by
        # frequency, so that we are doing \int S R nu^-1 dnu rather than
        # \int S R dnu.  However, to avoid underflow, normalize this a bit
        if not self._sens_energy:
            if self._nresp > 1:
                midfreq = self._freq[self._nresp // 2]
            else:
                midfreq = self._freq[0]
            self._sedmult *= (midfreq / self._freq)

        #And, now the normalization condition
        self._normtype = ntyp
        if ntyp == "none":
            self._normparam = None
            self._normfac = -1.0

            #We still need to compute the effective wavelength --
            # assume constant fnu SED for this
            eff_freq = (self._freq * self._sedmult).sum() / self._sedmult.sum()
            self._effective_freq = eff_freq
            self._effective_wave = 299792458e-3 / eff_freq

        else:
            if ntyp == "power":
                self._normparam = float(normparam)
                sed = (self._freq / self._normfreq)**self._normparam
            elif ntyp == "flat":
                sed = numpy.ones(self._nresp)
                self._normparam = None
            elif ntyp == "bb":
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

            #Get effective frequency
            eff_freq = (self._freq * sed * self._sedmult).sum() * self._normfac
            self._effective_freq = eff_freq
            self._effective_wave = 299792458e-3 / eff_freq


        def _setup_delta(self, val, xtyp, xun):
            """Sets up delta function response, a special case"""
            if val <= 0:
                raise ValueError("Non-positive value")
            if xtyp == 'wave':
                if xun == 'angstroms':
                    self._normwave = 1e-4 * val
                elif xun == 'microns':
                    self._normwave = val
                elif xun == 'meters':
                    self._normwave = 1e6 * val
                else:
                    raise ValueError("Unrecognized wavelength type %s" % xtype)
                self._normfreq = 299792458e-3 / self._normwave
            elif xtyp == 'freq':
                # Change to GHz
                if xun == 'hz':
                    self._normfreq = 1e-9 * val
                elif xun == 'mhz':
                    self._normfreq = 1e-3 * val
                elif xun == 'ghz':
                    self._normfreq = val
                elif xun == 'thz':
                    self._normfreq = 1e3 * val
                self._normwave = 299792458e-3 / self._normfreq
            self._isdelta = True
            self._resp = numpy.array([1.0])
            self._nresp = 1
            self._normtype = "delta"
            self._normparam = None
            self._normfac = 1.0
            self._effective_wave = self._normwave
            self._effective_freq = 299792458e-3 / self._effective_wave
            self._sens_energy = True
            return

    @property
    def name(self):
        return self._name

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
    def effective_wavelength(self):
        """ Get filter effective wavelength in microns"""
        return self._effective_wave

    @property
    def effective_frequency(self):
        """ Get filter effective frequency in GHz"""
        return self._effective_freq

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

    def __str__(self):
        return "name: %s lambda_eff: %0gum" % (self._name, self._effective_wave)

class response_set(object):
    """ A set of instrument responses"""
    
    def __init__(self, inputfile, dir=None):
        """ Initialize response set.

        Parameters
        ----------
        inputfile : string
          Name of input file
        """

        import astropy.io.ascii
        
        if not isinstance(inputfile, basestring):
            raise TypeError("filename must be string-like")
        
        if not dir is None:
            if not isinstance(dir, basestring):
                raise TypeError("dir must be string-like")

        
        if dir is None:
            data = astropy.io.ascii.read(inputfile, comment='^#')
        else:
            data = astropy.io.ascii.read(os.path.join(dir,inputfile), 
                                         comment='^#')
        if len(data) == 0 :
            raise IOError("No data read from %s" % inputfile)

        self._responses = {}
        for dat in data:
            name = dat[0]
            self._responses[name] = response(name, dat[1], dat[2].lower(),
                                             dat[3].lower(), dat[4].lower(),
                                             dat[5].lower(), float(dat[6]), 
                                             float(dat[7]), dir=dir)

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
