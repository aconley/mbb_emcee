import math
import numpy
from pkg_resources import resource_filename
import os.path
import astropy.io.ascii
import scipy.integrate
import h5py

"""Astronomical instrument response modeling"""

__all__ = ["response", "response_set"]

#hack for basestring
try:
    basestring
except:
    #Python 3
    basestring = str

special_types = ["delta", "box", "gauss", "alma"]


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
    h = 6.6260693e-34  # J/s
    k = 1.3806505e-23  # J/K
    hokt = 1e9 * h / (k * float(temperature))  # The 1e9 is GHz -> Hz

    return freq**3 / numpy.expm1(hokt * freq)


class response(object):
    """ A class representing the response of an instrument to an observation.

    Handles instrument responses and pipeline normalization conventions.
    """

    def __init__(self, name):
        """
        Parameters
        ----------
        name : string
          Brief name of response function
        """

        self._name = str(name)
        self._data_read = False

    def setup(self, inputspec, xtype='wave', xunits='microns',
              senstype='energy', normtype='power', xnorm=250.0,
              normparam=-1.0, dir=None):
        """ Set up the response function, usually by reading an input file

        Parameters
        ----------
        inputspec : string
          Name of input file.  Must be a text file.  If this has the special
          values delta_?, box_?, or gauss_?, then rather than reading from a
          file the passband is created internally.

        xtype : string
          Type of input x variable.  lambda for wavelengths, freq for
          frequency.

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
          Normalization parameter -- power low exponent for power-law,
          temperature for blackbody

        dir : string
          Directory to look for inputspec in.  Defaults to current directory.

        Notes
        -----
        If inputspec is delta_val, then the filter function is assumed to be a
        delta function at value.  normtype, etc. is ignored.  For example, if
        xunit is microns, then delta_880 sets up a delta function at 880um.
        If xunit were GHz, then it would be a delta function at 880GHz.

        If inputspec is box_val1_val2, then the filter function is a box with
        11 points centered on val1 and with a width of val2 (in the units of
        xunits)

        If inputspec is gauss_val1_val2, then the filter function is a
        Gaussian centered on val1 with a FWHM of val2.  It is sampled every
        FWHM/7 steps out to 3*FWHM in each direction.

        So don't name your input files delta_?, gauss_?, or box_?
        """

        # Make sure all args have right case
        ntyp = normtype.lower()
        xtyp = xtype.lower()
        xun = xunits.lower()
        styp = senstype.lower()

        # Read in or set-up the data
        # If there is a _, we may have a special case.
        # delta function allows quick return
        #Read in data or setup special case
        if not isinstance(inputspec, basestring):
            raise TypeError("filename must be string-like")
        self._isdelta = False
        if inputspec.find('_'):
            #First, check if it's a delta.  If so, do the
            # quick return setup for that
            spl = inputspec.split('_')
            bs = spl[0].lower()
            if bs == "delta":
                if len(spl) < 2:
                    raise ValueError("delta needs central frequency")
                val = float(spl[1])
                self._setup_delta(val, xtyp, xun)
                return
            elif bs == "box":
                if len(spl) < 3:
                    errstr = "box car needs 2 params in {:s}"
                    raise ValueError(errstr.format(inputspec))
                xvals, self._resp = self._setup_box(float(spl[1]),
                                                    float(spl[2]))
            elif bs == "gauss":
                if len(spl) < 3:
                    errstr = "gaussian needs 2 params in {:s}"
                    raise ValueError(errstr.format(inputspec))
                xvals, self._resp = self._setup_gauss(float(spl[1]),
                                                      float(spl[2]))

            elif bs == "alma":
                if len(spl) < 2:
                    errstr = "alma needs 1 params in {:s}".format(inputspec)
                    raise ValueError(errstr)
                xvals, self._resp = self._setup_alma(float(spl[1]),
                                                     xtyp, xun)
            else:
                # A real file -- read it!
                if dir is None:
                    infile = inputspec
                else:
                    infile = os.path.join(dir, inputspec)

                data = astropy.io.ascii.read(infile, comment='^#')
                if len(data) == 0:
                    raise IOError("No data read from {:s}".format(infile))

                xvals = numpy.asarray([dat[0] for dat in data])
                self._resp = numpy.asarray([dat[1] for dat in data])

        # We don't allow negative responses or xvals
        if xvals.min() <= 0:
            raise ValueError("Non-positive x value encountered")
        if self._resp.min() < 0:
            raise ValueError("Negative response encountered")
        if xnorm <= 0:
            raise ValueError("Non-positive xnorm")

        # Build up wavelength/normwave in microns, freq in GHz
        # This should be replaced with the use of astropy.units
        if xtyp == 'wave':
            if xun == 'angstroms' or xun == 'a':
                self._wave = 1e-4 * xvals
                self._normwave = 1e-4 * xnorm
            elif xun == 'microns' or xun == 'um':
                self._wave = xvals
                self._normwave = xnorm
            elif xun == 'meters' or xun == 'm':
                self._wave = 1e6 * xvals
                self._normwave = 1e6 * xnorm
            else:
                errmsg = "Unrecognized wavelength unit {:s}".format(xun)
                raise ValueError(errmsg)
            self._freq = 299792458e-3 / self._wave  # In GHz
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
            else:
                errmsg = "Unrecognized frequency unit {:s}".format(xun)
                raise ValueError(errmsg)

            self._wave = 299792458e-3 / self._freq  # Microns
            self._normwave = 299792458e-3 / self._normfreq

        # Sort into ascending wavelength order for simplicity
        idx = self._wave.argsort()
        self._wave = self._wave[idx]
        self._freq = self._freq[idx]
        self._resp = self._resp[idx]
        self._nresp = len(self._resp)

        # Unity normalize the response (this is arbitrary, but convenient
        # for display purposes)
        self._resp /= self._resp.max()

        # Set up sensitivity type
        if styp == "energy":
            self._sens_energy = True
        elif styp == "counts":
            self._sens_energy = False
        else:
            raise ValueError("Unknown sensitivity type {:s}".format(senstype))

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
        self._normtype = str(ntyp)
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
                    errmsg = "Invalid (non-positive) blackbody "\
                             "temperature {:f}"
                    raise ValueError(errmsg.format(self._normparam))
                # Blackbody curve, normalized at normfreq
                sed = response_bb(self._freq, self._normparam) / \
                    response_bb(self._normfreq, self._normparam)
            else:
                errmsg = "Unknown normalization type {:s}"
                raise ValueError(errmsg.format(normtype))

            # Note this will be negative due to the frequency ordering,
            # but that's okay because our normal integrals will be negative
            # as well so when we divide them it will all work out.
            self._normfac = 1.0 / (sed * self._sedmult).sum()

            #Get effective frequency
            eff_freq = (self._freq * sed * self._sedmult).sum() * self._normfac
            self._effective_freq = eff_freq
            self._effective_wave = 299792458e-3 / eff_freq

        self._data_read = True

    def _setup_delta(self, val, xtyp, xun):
        """Sets up delta function response, a special case"""
        if val <= 0:
            raise ValueError("Non-positive value")
        if xtyp == 'wave':
            if xun == 'angstroms' or xun == 'a':
                self._normwave = 1e-4 * val
            elif xun == 'microns' or xun == 'um':
                self._normwave = val
            elif xun == 'meters' or xun == 'm':
                self._normwave = 1e6 * val
            else:
                errmsg = "Unrecognized wavelength type {:s}"
                raise ValueError(errmsg.format(xtype))
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
        self._effective_wave = self._normwave
        self._effective_freq = 299792458e-3 / self._effective_wave
        self._wave = numpy.array([self._normwave])
        self._freq = numpy.array([self._effective_freq])
        self._resp = numpy.array([1.0])
        self._nresp = 1
        self._normtype = "delta"
        self._normparam = None
        self._normfac = 1.0
        self._sens_energy = True

    def _setup_box(self, cent, width):
        npoints = 11
        xvals = numpy.linspace(cent - 0.5 * width,
                               cent + 0.5 * width,
                               npoints)
        resp = numpy.ones(npoints)
        return xvals, resp

    def _setup_gauss(self, cent, fwhm):
        minval = cent - 3.0*fwhm
        maxval = cent + 3.0*fwhm
        npoints = 43  # (FWHM/7)
        sig = fwhm / math.sqrt(8 * math.log(2))
        xvals = numpy.linspace(cent - 3.0*fwhm,
                               cent + 3.0*fwhm,
                               npoints)
        resp = numpy.exp(-0.5 * ((xvals - cent) / sig)**2)
        return xvals, resp

    def _setup_alma(self, cent, xtype='freq', xunit='gHz'):
        # Get central frequency in GHz
        npoints = 13
        npoints0 = 3
        if xtype == 'wave':
            if xunit == 'angstroms':
                cen_freq = 2997924580.0 / cent
            elif xunit == 'microns':
                cen_freq = 299792458e-3 / cent
            elif xunit == 'meters':
                cen_freq = 299792458.0e-9 / cent
            else:
                errmsg = "Unrecognized wavelength unit {:s}".format(xunit)
                raise ValueError(errmsg)
        elif xtype == 'freq':
            if xunit == 'hz':
                cen_freq = cent * 1e-9
            elif xunit == 'mhz':
                cen_freq = cent * 1e-3
            elif xunit == 'ghz':
                cen_freq = cent
            elif xunit == 'thz':
                cen_freq = cent * 1e3
            else:
                errmsg = "Unrecognized frequency unit {:s}".format(xunit)
                raise ValueError(errmsg)
        else:
            raise ValueError("Unknown unit type {:s}".format(xtype))

        # Identify band: bands 3, 4, 6, 7, 8; don't support band 9 since
        # it's more complex (DSB vs. 2SB).  Bands 1, 2, 5, 10 not implemented
        if_low = numpy.array([92.0, 125, 221, 283, 385])
        if_high = numpy.array([108.0, 163, 265, 365, 500])
        if_range_bot = numpy.array([4.0, 4.0, 6, 4, 4])
        wband = numpy.nonzero((cen_freq >= if_low) & (cen_freq <= if_high))[0]
        if len(wband) == 0:
            errmsg = "Unable to identify ALMA band with central freq {:0.1f}"
            raise ValueError(errmsg.format(cen_freq))
        xvals0 = numpy.linspace(cen_freq - if_range_bot[wband] - 3.75,
                                cen_freq - if_range_bot[wband], npoints)
        xvals1 = numpy.linspace(cen_freq - if_range_bot[wband] + 0.0001,
                                cen_freq + if_range_bot[wband] - 0.0001,
                                npoints0)
        xvals2 = numpy.linspace(cen_freq + if_range_bot[wband],
                                cen_freq + if_range_bot[wband] + 3.75, npoints)
        xvals = numpy.concatenate((xvals0, xvals1, xvals2))
        resp = numpy.concatenate((numpy.ones(npoints), numpy.zeros(npoints0),
                                  numpy.ones(npoints)))
        return xvals, resp

    @property
    def name(self):
        return self._name

    @property
    def data_read(self):
        return self._data_read

    @property
    def wavelength(self):
        """ Wavelength of response function in microns"""
        if not self._data_read:
            return None
        return self._wave

    @property
    def frequency(self):
        """ Frequency of response function in GHz"""
        if not self._data_read:
            return None
        return self._freq

    @property
    def response(self):
        """ Response function at wavelengths specified by wave"""
        if not self._data_read:
            return None
        return self._resp

    @property
    def effective_wavelength(self):
        """ Get filter effective wavelength in microns"""
        if not self._data_read:
            return None
        return self._effective_wave

    @property
    def effective_frequency(self):
        """ Get filter effective frequency in GHz"""
        if not self._data_read:
            return None
        return self._effective_freq

    @property
    def normfac(self):
        """ Normalization value"""
        # Multiply by -1 to avoid confusion
        if not self._data_read:
            return None
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

        if not self._data_read:
            raise Exception("Data not read yet, can't get response")

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

    def writeToHDF5(self, handle):
        """ Writes the response to an HDF5 handle (file, group)"""

        if not self._data_read:
            raise ValueError("Data must be read to write as HDF5")

        handle.attrs["Name"] = self._name
        handle.attrs["IsDelta"] = self._isdelta
        handle.attrs["WaveUnits"] = "microns"
        handle.attrs["FreqUnits"] = "GHz"
        handle.attrs["NormWave"] = self._normwave
        handle.attrs["NormFreq"] = self._normfreq
        handle.attrs["NormType"] = self._normtype
        if not self._normparam is None:
            handle.attrs["NormParam"] = self._normparam
        handle.attrs["NormFac"] = self._normfac
        handle.attrs["SensEnergy"] = self._sens_energy
        handle.attrs["EffectiveFreq"] = self._effective_freq
        handle.attrs["EffectiveWave"] = self._effective_wave
        handle.attrs["NResp"] = self._nresp
        handle.create_dataset("Wave", data=self._wave)
        handle.create_dataset("Freq", data=self._freq)
        handle.create_dataset("Response", data=self._resp)
        if not self._isdelta:
            handle.create_dataset("Dnu", data=self._dnu)
            handle.create_dataset("Sedmult", data=self._sedmult)

    def readFromHDF5(self, handle):
        """ Reads the response to an HDF5 handle (file, group)"""

        self._name = handle.attrs["Name"]
        self._isdelta = handle.attrs["IsDelta"]
        self._normwave = handle.attrs["NormWave"]
        self._normfreq = handle.attrs["NormFreq"]
        self._normtype = handle.attrs["NormType"]
        if "NormParam" in handle.attrs:
            self._normparam = handle.attrs["NormParam"]
        else:
            self._normparam = None
        self._normfac = handle.attrs["NormFac"]
        self._sens_energy = handle.attrs["SensEnergy"]
        self._effective_freq = handle.attrs["EffectiveFreq"]
        self._effective_wave = handle.attrs["EffectiveWave"]
        self._nresp = handle.attrs["NResp"]
        self._wave = handle["Wave"][...]
        self._freq = handle["Freq"][...]
        self._resp = handle["Response"][...]
        if "Dnu" in handle:
            self._dnu = handle["Dnu"][...]
        else:
            if hasattr(self, "_dnu"):
                del self._dnu
        if "Sedmult" in handle:
            self._sedmult = handle["Sedmult"][...]
        else:
            if hasattr(self, "_sedmult"):
                del self._sedmult
        self._data_read = True

    def __str__(self):
        val = "{0:s} lambda_eff: {1:0.1f} [um]"
        return val.format(self._name, self._effective_wave)


class response_set(object):
    """ A set of instrument responses."""

    def __init__(self, inputfile=None, dir=None):
        """ Initialize response set.

        Parameters
        ----------
        inputfile: string
          Name of input file

        dir: string
          Directory to look for responses in; defaults to built-in location
        """

        self._responses = {}
        # Note this loads a default if inputfile is None
        self.read(inputfile=inputfile, dir=dir)

    def read(self, inputfile=None, dir=None):
        """ Read in responses

        Parameters
        ----------
        inputfile : string
          Name of input file.  If None, defaults to built-in location

        dir: string
          Directory to look for responses in; defaults to current directory,
          but is ignored if using default inputfile

        Notes
        -----
          This will clear out all previously loaded information.
        """

        import astropy.io.ascii

        if inputfile is None:
            infile = resource_filename(__name__,
                                       'resources/mbb_filterwheel.txt')
            indir = resource_filename(__name__, 'resources/')
        elif not isinstance(inputfile, basestring):
            raise TypeError("filename must be string-like")
        else:
            if dir is None:
                infile = inputfile
                indir = None
            else:
                if not isinstance(dir, basestring):
                    raise TypeError("dir must be string-like")
                infile = os.path.join(dir, inputfile)
                indir = dir

        data = astropy.io.ascii.read(infile, comment='^#')
        if len(data) == 0:
            raise IOError("No data read from {:s}".format(inputfile))

        # Clear old responses
        self._responses.clear()

        # Read in all the responses
        for dat in data:
            self.add(dat[0], dat[1], dat[2].lower(), dat[3].lower(),
                     dat[4].lower(), dat[5].lower(), float(dat[6]),
                     float(dat[7]), dir=indir)

    def add(self, name, spec, xtype, xunits, senstype, normtype, xnorm,
            normparam, dir=dir):
        """ Add a filter."""

        resp = response(name)
        resp.setup(spec, xtype=xtype, xunits=xunits, senstype=senstype,
                   normtype=normtype, xnorm=xnorm, normparam=normparam,
                   dir=dir)
        self._responses[name] = resp

    def add_special(self, name):
        """ Add a 'special' filter (boxcar, delta, gaussian, alma, etc.).

        This is a shortcut for adding special filters (boxcar, gaussian,
        etc.) on the fly.  The language is related to that used by setup,
        but altered a bit because only a single input is provided
        (the specification).  As a result, the sensitivity is always
        assumed to be energy and the normalization flat.

        The format is name_type_values, where type is the special type
        (box, gauss, etc.), and values are the things needed to specify
        the type.  On the first value it is possible to specify units
        (GHz or um) on the first argument -- so ZSpec_box_1050um_100
        will make a boxcar in wavelength space, but SMA_box_234_16Ghz_8
        will make one in frequency space.  The default is frequency
        space (in GHz), since the most common use should be processing
        interferometric observations.
        """

        import re

        # Make sure the lead in is one of the recognized types
        spl = name.split('_')
        bs = spl[1].lower()
        if not bs in special_types:
            errmsg = "Unknown 'special' response type in {:s}"
            raise ValueError(errmsg.format(name))

        # Look for units on first argument
        if len(spl) < 3:
            raise ValueError("Special type has no numerical spec")
        firstarg = spl[2].lower()
        num = re.findall(r'\d+', firstarg)
        if len(num) == 0:
            raise ValueError("Special type needs numeric specification")
        p = re.compile(r'^\d+')
        firstarg_unit = p.sub("", firstarg)
        if len(firstarg_unit) == 0:
            xtype = 'freq'
            xunit = 'ghz'
        elif firstarg_unit == 'ghz':
            xtype = 'freq'
            xunit = 'ghz'
        elif firstarg_unit == 'um':
            xtype = 'wave'
            xunit = 'microns'
        else:
            errstr = "Unable to understand unit specification {:s}"
            raise ValueError(errstr.format(firstarg_unit))

        # We have to build a setup stype spec
        newspec = bs + '_' + num[0]
        if len(spl) > 3:
            newspec += '_' + '_'.join(spl[3:])

        resp = response(name)
        resp.setup(newspec, xtype=xtype, xunits=xunit, senstype='energy',
                   normtype='flat', xnorm=float(num[0]), normparam=0)
        self._responses[name] = resp

    def writeToHDF5(self, handle):
        """ Writes the response sets to an HDF5 handle (file, group).

        Each response goes into it's own group.
        """

        for name in self._responses:
            d = handle.create_group(name)
            self._responses[name].writeToHDF5(d)

    def readFromHDF5(self, handle):
        """ Reads the response sets from a HDF5 file"""

        # Clear old ones
        self._responses.clear()

        for name in handle:
            if type(handle[name]) == h5py._hl.group.Group:
                resp = response(name)
                resp.readFromHDF5(handle[name])
                self._responses[name] = resp

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

    def keys(self):
        """ Available responses"""
        return self._responses.keys()

    # in
    def __contains__(self, val):
        """ Is a particular response available?"""
        return val in self._responses

    def items(self):
        """ Get all responses"""
        return self._responses.items()

    def values(self):
        """ Get all responses"""
        return self._responses.values()

    def delitem(self, val):
        del self._responses[val]

    def __str__(self):
        return '\n'.join([str(self._responses[nm]) for nm in self._responses])
