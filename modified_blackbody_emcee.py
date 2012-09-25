#!/usr/bin/env python

"""MCMC model for modified blackbody fit to far-IR/sub-mm/mm photometry.
This works in the observer frame."""

#Requires that numpy, scipy, emcee, asciitable, pyfits are installed

import numpy
import emcee
import math
from modified_blackbody_like import modified_blackbody_like

if __name__ == "__main__" :
    #Do the fit!
    import argparse
    import os.path
    import pickle
    import textwrap

    desc = """Fit a modified blackbody to user provided data using an MCMC."""


    epi = textwrap.dedent('''
    The fit is done using rest frame quantities.  Input wavelengths
    should be specified in microns and fluxes in mJy.

    The model is (schematically)

      S_nu = (1-exp(-tau)) B_nu,

    where B_nu is the Planck function, and tau is the optical depth, assumed
    of the form

      tau = (nu/nu0)^beta.

    The fit parameters are:
      T:  The observer frame temperature in [K] (Trest/(1+z))

      beta: The dust attenuation slope

      lambda0: The observer frame wavlength where the dust becomes optically
               thick (lambda0rest * (1+z)) in [um]

      alpha: The blue side power law slope

      fnorm: The normalization flux in [mJy] in the observer frame (usually 
             at 500 [um], but this can be changed with --wavenorm)
     ''')

    parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('photfile',action='store',
                        help="Text file holding photometry in microns, mJy, error")
    parser.add_argument('outfile',action='store',
                        help="File to pickle resulting chain to")
    parser.add_argument('-b','--burn',action='store',type=int,default=500,
                        help="Number of burn-in steps to do (def: 500)")
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
    parser.add_argument('-k','--kappanu',action='store',type=float,
                        default=2.64,
                        help="Dust mass coefficient in m^2 kg^-1 (def: 2.64)")
    parser.add_argument("--kappalambda",action="store",type=float,default=125.0,
                        help="Rest frame wavelength (in um) at which kappanu is defined (def: 125)")
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
    parser.add_argument('-n','--nwalkers',action='store',type=int,
                        help="Number of walkers to use in MCMC (Def: 250)",
                        default=250)
    parser.add_argument('-N','--nsteps',action='store',type=int,
                        default=1000,
                        help="Number of steps to take per walker (Def: 1000)")
    parser.add_argument('--noalpha',action='store_true',default=False,
                        help="Do not use blue side power law in fit")
    parser.add_argument('--nlir',action="store",type=int,
                        default=None,
                        help="Number of steps to include in LIR, dust mass calculations (Def: All of them)")
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
                        version='%(prog)s 1.0')
    parser.add_argument('-w','--wavenorm',action='store', 
                        type=float, default=500.0,
                        help="Observer frame wavelength of normalization (def: 500)")

    results = parser.parse_args() #Runs on sys.argv by default

    if not results.photdir is None :
        photfile = os.path.join(results.photdir,results.photfile)
    else :
        photfile = results.photfile
        
    if not results.covfile is None :
        if not results.photdir is None :
            covfile = os.path.join(results.photdir,results.covfile)
        else : covfile = results.covfile
    else: covfile = None

    #This object handles all the calculations
    sed = modified_blackbody_like(photfile,covfile=covfile,
                                  covextn=results.covextn, 
                                  wavenorm=results.wavenorm,
                                  redshift=results.redshift,
                                  noalpha=results.noalpha,
                                  opthin=results.opthin)

    
    #Set parameters fixed/limits if present
    if results.fixT: sed.fix_param(0)
    if results.fixBeta: sed.fix_param(1)
    if results.fixLambda0: sed.fix_param(2)
    if results.noalpha or results.fixAlpha:
        sed.fix_param(3)
    if results.fixFnorm: sed.fix_param(4)
        
    #Lower limits
    if not results.lowT is None: 
        sed.set_lowlim(0,results.lowT)
    if not results.lowBeta is None: 
        sed.set_lowlim(1,results.lowBeta)
    if (not results.opthin) and (not results.lowLambda0 is None): 
        sed.set_lowlim(2,results.lowLambda0)
    if (not results.noalpha) and (not results.lowAlpha is None): 
        sed.set_lowlim(3,results.lowAlpha)
    if not results.lowFnorm is None: 
        sed.set_lowlim(4,results.lowFnorm)

    #Upper Limits
    if not results.upT is None: 
        sed.set_uplim(0,results.upT)
    if not results.upBeta is None : 
        sed.set_uplim(1,results.upBeta)
    if (not results.opthin) and (not results.upLambda0 is None): 
        sed.set_uplim(2,results.upLambda0)
    if (not results.noalpha) and (not results.upAlpha is None): 
        sed.set_uplim(3,results.upAlpha)
    if not results.upFnorm is None : 
        sed.set_uplim(4,results.upFnorm)

    #Priors
    if not results.priorT is None:
        sed.set_gaussian_prior(0, results.priorT[0], results.priorT[1])
    if not results.priorBeta is None:
        sed.set_gaussian_prior(1, results.priorBeta[0], results.priorBeta[1])
    if (not results.opthin) and (not results.priorLambda0 is None):
        sed.set_gaussian_prior(2, results.priorLambda0[0], 
                               results.priorLambda0[1])
    if (not results.noalpha) and (not results.priorT is None):
        sed.set_gaussian_prior(3, results.priorAlpha[0], results.priorAlpha[1])
    if not results.priorFnorm is None:
        sed.set_gaussian_prior(4, results.priorFnorm[0], results.priorFnorm[1])

    #Set up sampler
    nwalkers = results.nwalkers
    if (nwalkers <= 0) :
        raise ValueError("Invalid (non-positive) nwalkers: %d" % nwalkers)

    #initial values -- T, beta, lambda0, alpha, fnorm
    #Should add way for user to specify these
    p0mn  = numpy.array([results.initT, results.initBeta, 
                         results.initLambda0, results.initAlpha,
                         results.initFnorm])
    p0var = numpy.array([2, 0.2, 100, 0.3, 5.0])
    if results.fixT:
        p0var[0] = 0.0
    if results.fixBeta:
        p0var[1] = 0.0
    if results.opthin or results.fixLambda0:
        p0var[2] = 0.0
    if results.noalpha or results.fixAlpha:
        p0var[3] = 0.0
    if results.fixFnorm:
        p0var[4] = 0.0
    npar = len(p0mn)
    p0 = [ numpy.random.randn(npar)*p0var + p0mn for i in xrange(nwalkers) ]

    #Create sampler
    sampler = emcee.EnsembleSampler(nwalkers, npar, sed,
                                    threads=results.threads)

    #Do burn-in
    if results.burn <= 0 :
        raise ValueError("Invalid (non-positive) number of burn in steps: %d" %
                         results.burn)
    if results.verbose : print "Doing burn in with %d steps" % results.burn
    pos, prob, state = sampler.run_mcmc(p0, results.burn)
    
    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    if results.nsteps <= 0 :
        raise ValueError("Invalid (non-positive) number of steps: %d" % 
                         results.nsteps)
    if results.verbose : print "Main chain with %d steps" % results.nsteps
    st = sampler.run_mcmc(pos, results.nsteps, rstate0=state)

    if results.verbose :
        print "Mean acceptance fraction:", \
            numpy.mean(sampler.acceptance_fraction)
        try :
            acor = sampler.acor
            print "Autocorrelation time: "
            print " Number of steps %d should be larger than this" % \
                results.nsteps
            print "\tT:        %f" % acor[0]
            print "\tbeta:     %f" % acor[1]
            if not results.noalpha:
                print "\talpha:    %f" % acor[2]
            print "\tlambda0:  %f" % acor[3]
            print "\tfnorm:    %f" % acor[4]
        except ImportError :
            print "Unable to estimate autocorrelation time (acor not installed)"


    #Save results
    if results.verbose : print "Saving results"
    output = open(results.outfile,'wb')
    chn = sampler.flatchain
    pickle.dump(chn,output)
    output.close()
