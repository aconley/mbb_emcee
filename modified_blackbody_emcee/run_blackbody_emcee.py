#!python

"""MCMC model for modified blackbody fit to far-IR/sub-mm/mm photometry.
This works in the observer frame."""

#Requires that numpy, scipy, emcee, asciitable, pyfits are installed

import numpy
from modified_blackbody_emcee import modified_blackbody_fit
from modified_blackbody_emcee import modified_blackbody_results

if __name__ == "__main__":
    import argparse
    import os.path
    import pickle
    import textwrap

    desc = """Fit a modified blackbody to user provided data using an MCMC."""


    epi = textwrap.dedent('''
    The fit is done using rest frame quantities.  Input wavelengths
    should be specified in microns and fluxes in mJy.

    The model is (schematically)

      S_nu \propto (1-exp(-tau)) B_nu,

    where B_nu is the Planck function, and tau is the optical depth, assumed
    of the form

      tau = (nu/nu0)^beta.

    The fit parameters are (in this order):
      T:  The observer frame temperature in [K] (Trest/(1+z))

      beta: The dust attenuation slope

      lambda0: The observer frame wavlength where the dust becomes optically
               thick (lambda0rest * (1+z)) in [um]

      alpha: The blue side power law slope

      fnorm: The normalization flux in [mJy] in the observer frame (usually 
             at 500 [um], but this can be changed with --wavenorm)

      Note that alpha and beta always have upper limits of 20, since larger 
      values frequently cause overflows.  Lambda0 has an upper limit of
      10 times the longest wavelength data point.  The user can override
      all these limits, but not turn them off.
     ''')

    parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('photfile',action='store',
                    help="Text file holding photometry in microns, mJy, error")
    parser.add_argument('outfile',action='store',
                        help="File to pickle resulting chain to")
    parser.add_argument('-b','--burn',action='store',type=int,default=50,
                        help="Number of burn-in steps to do (def: 50)")
    parser.add_argument('-c','--covfile',action='store',
                        help="FITS file containing covariances (in mJy)",
                        default=None)
    parser.add_argument('-e','--covextn',action='store',default=0,type=int,
                        help="Extension of FITS file to look for cov matrix in (Def: 0)")
    parser.add_argument('--fixT',action='store_true', default=None,
                        help="Fix T to initial value")
    parser.add_argument('--fixBeta',action='store_true', default=None,
                        help="Fix Beta to initial value")
    parser.add_argument('--fixAlpha',action='store_true', default=None,
                        help="Fix Alpha to initial value")
    parser.add_argument('--fixLambda0',action='store_true', default=None,
                        help="Fix Lambda0 to initial value")
    parser.add_argument('--fixFnorm',action='store_true', default=None,
                        help="Fix Fnorm to initial value")
    parser.add_argument('--initT', action='store', type=float, default=10.0,
                        help="Initial T/(1+z)")
    parser.add_argument('--initBeta', action='store', type=float, default=2.0,
                        help="Initial beta")
    parser.add_argument('--initAlpha', action='store', type=float, default=4.0,
                        help="Initial alpha")
    parser.add_argument('--initLambda0', action='store', type=float, 
                        default=2500.0,
                        help="Initial Lambda0*(1+z)")
    parser.add_argument('--initFnorm', action='store', type=float, default=40.0,
                        help="Initial Fnorm")
    parser.add_argument('--get_dustmass',action='store_true',default=False,
                        help="Estimate dust mass in 10^8 M_sun.  You must set the redshift")
    parser.add_argument('--get_lagn',action='store_true',default=False,
                        help="Get the rest frame 42.5 to 122.5um lumiosity.  You must set the redshift")
    parser.add_argument('--get_lir',action='store_true',default=False,
                        help="Get the rest frame 8-1000um luminosity.  You must set the redshift")
    parser.add_argument('--get_peaklambda', action='store_true', default=False,
                        help="Get the observer frame SED peak wavelength")
    parser.add_argument("--lowT",action='store',type=float,default=None,
                        help="Lower limit on T (Def:0)")
    parser.add_argument("--lowBeta",action='store',type=float,default=None,
                        help="Lower limit on beta (Def:0)")
    parser.add_argument("--lowLambda0",action='store',type=float,default=None,
                        help="Lower limit on lambda0 (Def:0)")
    parser.add_argument("--lowAlpha",action='store',type=float,default=None,
                        help="Lower limit on alpha (Def:0)")
    parser.add_argument("--lowFnorm",action='store',type=float,default=None,
                        help="Lower limit on fnorm (Def:0)")
    parser.add_argument('-p','--photdir',action='store',
                        help="Directory to look for files in",
                        default=None)
    parser.add_argument('--maxidx',action='store',type=int,default=None,
                        help="Maximum number of steps (per walker) to include in L_IR, L_AGN, and dustmass")
    parser.add_argument('-n','--nwalkers',action='store',type=int,
                        help="Number of walkers to use in MCMC (Def: 250)",
                        default=250)
    parser.add_argument('-N','--nsteps',action='store',type=int,
                        default=250,
                        help="Number of steps to take per walker (Def: 250)")
    parser.add_argument('--noalpha',action='store_true',default=False,
                        help="Do not use blue side power law in fit")
    parser.add_argument('--opthin',action="store_true",default=False,
                        help="Assume optically thin model")
    parser.add_argument('--priorT',action="store",nargs=2,type=float,
                        default=None,help="Mean and sigma of T prior")
    parser.add_argument('--priorBeta',action="store",nargs=2,type=float,
                        default=None,help="Mean and sigma of Beta prior")
    parser.add_argument('--priorLambda0',action="store",nargs=2,type=float,
                        default=None,help="Mean and sigma of Lambda0 prior")
    parser.add_argument('--priorAlpha',action="store",nargs=2,type=float,
                        default=None,help="Mean and sigma of Alpha prior")
    parser.add_argument('--priorFnorm',action="store",nargs=2,type=float,
                        default=None,help="Mean and sigma of fnorm prior")
    parser.add_argument('-t','--threads',action='store',type=int,default=1,
                        help="Number of threads to use (Def: 1)")
    parser.add_argument("--upT",action='store',type=float,default=None,
                        help="Upper limit on T")
    parser.add_argument("--upBeta",action='store',type=float,default=None,
                        help="Upper limit on beta")
    parser.add_argument("--upAlpha",action='store',type=float,default=None,
                        help="Upper limit on alpha")
    parser.add_argument("--upLambda0",action='store',type=float,default=None,
                        help="Upper limit on lambda0")
    parser.add_argument("--upFnorm",action='store',type=float,default=None,
                        help="Upper limit on fnorm")
    parser.add_argument("-v","--verbose",action="store_true",default=False,
                        help="Print status messages")
    parser.add_argument('-V','--version',action='version',
                        version='%(prog)s 0.1.2')
    parser.add_argument('-w','--wavenorm',action='store', 
                        type=float, default=500.0,
                        help="Observer frame wavelength of normalization (def: 500)")
    parser.add_argument('-z','--redshift',action='store',
                        type=float, default=None,
                        help="Redshift of object")
    

    parse_results = parser.parse_args() #Runs on sys.argv by default

    # Set up fit
    npar = 5
    if not parse_results.photdir is None :
        photfile = os.path.join(parse_results.photdir,parse_results.photfile)
    else :
        photfile = parse_results.photfile
    
    if not parse_results.covfile is None :
        if not parse_results.photdir is None :
            covfile = os.path.join(parse_results.photdir,parse_results.covfile)
        else : covfile = parse_results.covfile
    else: covfile = None
    
    # Set up sampler
    nwalkers = parse_results.nwalkers
    if (nwalkers <= 0) :
        raise ValueError("Invalid (non-positive) nwalkers: %d" % nwalkers)

    fit = modified_blackbody_fit(photfile, covfile, parse_results.covextn,
                                 parse_results.wavenorm, parse_results.noalpha,
                                 parse_results.opthin, nwalkers, npar,
                                 parse_results.threads) 
    
    # Set parameters fixed/limits if present
    if parse_results.fixT: fit.like.fix_param(0)
    if parse_results.fixBeta: fit.like.fix_param(1)
    if parse_results.fixLambda0: fit.like.fix_param(2)
    if parse_results.noalpha or parse_results.fixAlpha:
        fit.like.fix_param(3)
        if parse_results.fixFnorm: fit.like.fix_param(4)
        
    # Lower limits
    if not parse_results.lowT is None: 
        fit.like.set_lowlim(0,parse_results.lowT)
    if not parse_results.lowBeta is None: 
        fit.like.set_lowlim(1,parse_results.lowBeta)
    if (not parse_results.opthin) and (not parse_results.lowLambda0 is None): 
        fit.like.set_lowlim(2,parse_results.lowLambda0)
    if (not parse_results.noalpha) and (not parse_results.lowAlpha is None): 
        fit.like.set_lowlim(3,parse_results.lowAlpha)
    if not parse_results.lowFnorm is None: 
        fit.like.set_lowlim(4,parse_results.lowFnorm)

    # Upper Limits
    if not parse_results.upT is None: 
        fit.like.set_uplim(0,parse_results.upT)
    if not parse_results.upBeta is None : 
        fit.like.set_uplim(1,parse_results.upBeta)
    if (not parse_results.opthin) and (not parse_results.upLambda0 is None): 
        fit.like.set_uplim(2,parse_results.upLambda0)
    if (not parse_results.noalpha) and (not parse_results.upAlpha is None): 
        fit.like.set_uplim(3,parse_results.upAlpha)
    if not parse_results.upFnorm is None : 
        fit.like.set_uplim(4,parse_results.upFnorm)

    # Priors
    if not parse_results.priorT is None:
        fit.like.set_gaussian_prior(0, parse_results.priorT[0], 
                                                  parse_results.priorT[1])
    if not parse_results.priorBeta is None:
        fit.like.set_gaussian_prior(1, parse_results.priorBeta[0],
                                                  parse_results.priorBeta[1])
    if (not parse_results.opthin) and (not parse_results.priorLambda0 is None):
        fit.like.set_gaussian_prior(2, parse_results.priorLambda0[0], 
                                                  parse_results.priorLambda0[1])
    if (not parse_results.noalpha) and (not parse_results.priorT is None):
        fit.like.set_gaussian_prior(3, parse_results.priorAlpha[0], 
                                                  parse_results.priorAlpha[1])
    if not parse_results.priorFnorm is None:
        fit.like.set_gaussian_prior(4, parse_results.priorFnorm[0], 
                                                  parse_results.priorFnorm[1])


    # initial values -- T, beta, lambda0, alpha, fnorm
    # Should add way for user to specify these
    p0mn  = numpy.array([parse_results.initT, parse_results.initBeta, 
                         parse_results.initLambda0, parse_results.initAlpha,
                         parse_results.initFnorm])
    p0var = numpy.array([2, 0.2, 100, 0.3, 5.0])
    if parse_results.fixT:
        p0var[0] = 0.0
    if parse_results.fixBeta:
        p0var[1] = 0.0
    if parse_results.opthin or parse_results.fixLambda0:
        p0var[2] = 0.0
    if parse_results.noalpha or parse_results.fixAlpha:
        p0var[3] = 0.0
    if parse_results.fixFnorm:
        p0var[4] = 0.0

    # We have to ensure all the initial values fall above the lower
    # limits
    p0 = []
    lowarr = fit.like._lowlim
    for i in range(nwalkers):
        pval = numpy.random.randn(npar)*p0var + p0mn
        badidx = pval < lowarr
        pval[badidx] = lowarr[badidx]
        p0.append(pval)


    # Do fit
    fit.run(parse_results.burn, parse_results.nsteps, p0, parse_results.verbose)

    # Peak wavelength computation
    if parse_results.get_peaklambda:
        if parse_results.verbose: print "Computing peak obs-frame wavelength"
        fit.get_peaklambda()

    # L_IR computation
    if parse_results.get_lir: 
        if parse_results.redshift is None:
            raise ValueError("Must provide redshift if computing L_IR")
        if parse_results.verbose: print "Computing L_IR (8-1000)"
        fit.get_lir(parse_results.redshift, maxidx=parse_results.maxidx)


    # L_AGN computation
    if parse_results.get_lagn: 
        if parse_results.redshift is None:
            raise ValueError("Must provide redshift if computing L_AGN")
        if parse_results.verbose: print "Computing L_AGN (42.5-122.5)"
        fit.get_lagn(parse_results.redshift, maxidx=parse_results.maxidx)

    # M_dust computation
    if parse_results.get_dustmass: 
        if parse_results.redshift is None:
            raise ValueError("Must provide redshift if computing m_dust")
        if parse_results.verbose: print "Computing dust mass"
        fit.get_dustmass(parse_results.redshift, maxidx=parse_results.maxidx)


    # Jam parse_results into a output struct, and save that
    if parse_results.verbose : print "Saving results"
    with open(parse_results.outfile,'wb') as fl:
        res = modified_blackbody_results(fit)
        pickle.dump(res, fl)
