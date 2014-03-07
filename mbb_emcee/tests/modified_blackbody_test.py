""" Unit tests for modified_blackbody"""
import numpy as np
from numpy.testing import assert_allclose
from mbb_emcee import modified_blackbody


def test_thick():
    """ Test thick modified blackbody"""

    mbb = modified_blackbody(10.0, 2.0, 800.0, 2.0, 45.0)
    assert_allclose(mbb.T, 10.0, atol=1e-4)
    assert_allclose(mbb.beta, 2.0, atol=1e-4)
    assert_allclose(mbb.lambda0, 800, atol=1e-2)
    assert_allclose(mbb.alpha, 2.0, atol=1e-4)
    assert_allclose(mbb.fnorm, 45.0, atol=1e-4)
    assert mbb.has_alpha
    assert not mbb.optically_thin
    assert_allclose(mbb(500), 45.0, atol=1e-4)
    wave = np.array([250.0, 350.0, 500.0, 850.0])
    expval = np.array([21.96268738, 39.53249977, 45.0, 22.06274444])
    assert_allclose(mbb(wave), expval, rtol=1e-4)


def test_thin():
    """ Test thin modified blackbody"""

    mbb = modified_blackbody(15.0, 1.8, 200.0, 3.0, 50.0, opthin=True)
    assert (mbb.T - 15.0) < 1e-4
    assert_allclose(mbb.T, 15.0, atol=1e-4)
    assert_allclose(mbb.beta, 1.8, atol=1e-4)
    assert_allclose(mbb.alpha, 3.0, atol=1e-4)
    assert_allclose(mbb.fnorm, 50.0, atol=1e-4)
    assert mbb.has_alpha
    assert mbb.optically_thin
    wave = np.array([250.0, 350.0, 500.0, 850.0])
    expval = np.array([178.34976, 111.03026, 50.0, 10.880588])
    assert_allclose(mbb(wave), expval, rtol=1e-4)


def test_thinthick_compare():
    """ Test to make sure thin/thick are similar for lambda >> lambda0"""

    mbb_thin = modified_blackbody(15.0, 1.8, 5.0, 3.0, 50.0,
                                  opthin=True)
    mbb_thick = modified_blackbody(15.0, 1.8, 5.0, 3.0, 50.0,
                                   opthin=False)
    wave = np.array([500.0, 850.0, 1100.0, 2500.0])
    assert_allclose(mbb_thin(wave), mbb_thick(wave), rtol=1e-3)


def test_merge():
    """ Test alpha law merge"""

    # Comparison values computed with Mathematica NSolve
    # Start with optically thin
    mbb = modified_blackbody(20.0, 1.9, None, 3.5, 50.0, noalpha=True,
                             opthin=True)
    assert mbb.wavemerge is None
    mbb = modified_blackbody(20.0, 1.9, None, 3.5, 50.0, opthin=True)
    assert_allclose(mbb.wavemerge, 85.66065, rtol=1e-3)
    mbb = modified_blackbody(35.0, 2.2, None, 2.8, 50.0, opthin=True)
    assert_allclose(mbb.wavemerge, 51.40211, rtol=1e-3)

    # More difficult optically thick case
    mbb = modified_blackbody(20.0, 1.9, 250.0, 3.5, 50.0, noalpha=True)
    assert mbb.wavemerge is None
    mbb = modified_blackbody(20.0, 1.9, 250.0, 3.5, 50.0)
    assert_allclose(mbb.wavemerge, 109.5506829, rtol=1e-3)
    mbb = modified_blackbody(40.0, 1.5, 600.0, 3.0, 50.0)
    assert_allclose(mbb.wavemerge, 60.10021595, rtol=1e-3)
