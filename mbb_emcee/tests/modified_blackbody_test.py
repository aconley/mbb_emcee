""" Unit tests for modified_blackbody"""
import unittest
import numpy as np
from mbb_emcee import modified_blackbody


class MBB(unittest.TestCase):

    def testNormalCase(self):
        """ Test thick modified blackbody"""

        mbb = modified_blackbody(10.0, 2.0, 800.0, 2.0, 45.0)
        self.assertAlmostEqual(mbb.T, 10.0, delta=1e-4)
        self.assertAlmostEqual(mbb.beta, 2.0, delta=1e-4)
        self.assertAlmostEqual(mbb.lambda0, 800, delta=1e-2)
        self.assertAlmostEqual(mbb.alpha, 2.0, delta=1e-4)
        self.assertAlmostEqual(mbb.fnorm, 45.0, delta=1e-4)
        self.assertTrue(mbb.has_alpha)
        self.assertFalse(mbb.optically_thin)
        self.assertAlmostEqual(mbb(500), 45.0, delta=1e-4)
        wave = np.array([250.0, 350.0, 500.0, 850.0])
        expval = np.array([21.96268738, 39.53249977, 45.0, 22.06274444])
        self.assertTrue(np.allclose(mbb(wave), expval, rtol=1e-4))

    def testOpthinCase(self):
        """ Test thin modified blackbody"""

        mbb = modified_blackbody(15.0, 1.8, 200.0, 3.0, 50.0, opthin=True)
        isclose = abs(mbb.T - 15.0) < 1e-4
        self.assertAlmostEqual(mbb.T, 15.0, delta=1e-4)
        self.assertAlmostEqual(mbb.beta, 1.8, delta=1e-4)
        self.assertAlmostEqual(mbb.alpha, 3.0, delta=1e-4)
        self.assertAlmostEqual(mbb.fnorm, 50.0, delta=1e-4)
        self.assertTrue(mbb.has_alpha)
        self.assertTrue(mbb.optically_thin)
        wave = np.array([250.0, 350.0, 500.0, 850.0])
        expval = np.array([178.34976, 111.03026, 50.0, 10.880588])
        self.assertTrue(np.allclose(mbb(wave), expval, rtol=1e-4))

    def testCompThinThick(self):
        """ Test to make sure thin/thick are similar for lambda >> lambda0"""

        mbb_thin = modified_blackbody(15.0, 1.8, 5.0, 3.0, 50.0,
                                      opthin=True)
        mbb_thick = modified_blackbody(15.0, 1.8, 5.0, 3.0, 50.0,
                                       opthin=False)
        wave = np.array([500.0, 850.0, 1100.0, 2500.0])
        self.assertTrue(np.allclose(mbb_thin(wave), mbb_thick(wave),
                                    rtol=1e-3))

    def testMerge(self):
        """ Test alpha law merge"""

        # Comparison values computed with Mathematica NSolve
        # Start with optically thin
        mbb = modified_blackbody(20.0, 1.9, None, 3.5, 50.0, noalpha=True,
                                 opthin=True)
        self.assertIsNone(mbb.wavemerge)
        mbb = modified_blackbody(20.0, 1.9, None, 3.5, 50.0, opthin=True)
        self.assertTrue(np.allclose(mbb.wavemerge, 85.66065, rtol=1e-3))
        mbb = modified_blackbody(35.0, 2.2, None, 2.8, 50.0, opthin=True)
        self.assertTrue(np.allclose(mbb.wavemerge, 51.40211, rtol=1e-3))

        # More difficult optically thick case
        mbb = modified_blackbody(20.0, 1.9, 250.0, 3.5, 50.0, noalpha=True)
        self.assertIsNone(mbb.wavemerge)
        mbb = modified_blackbody(20.0, 1.9, 250.0, 3.5, 50.0)
        self.assertTrue(np.allclose(mbb.wavemerge, 109.5506829, rtol=1e-3))
        mbb = modified_blackbody(40.0, 1.5, 600.0, 3.0, 50.0)
        self.assertTrue(np.allclose(mbb.wavemerge, 60.10021595, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
