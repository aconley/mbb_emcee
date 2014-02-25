""" Unit tests for response"""
import unittest
from mbb_emcee import response, response_set


class resptest(unittest.TestCase):

    def setUp(self):
        self.filter_wheel = response_set()

    def testDefaultMember(self):
        """ Test whether default set has expected members"""

        self.assertTrue("SCUBA2_850um" in self.filter_wheel)
        self.assertTrue("SPIRE_250um" in self.filter_wheel)
        self.assertTrue("SPIRE_350um" in self.filter_wheel)
        self.assertTrue("SPIRE_500um" in self.filter_wheel)
        self.assertTrue("Bolocam_1.1mm" in self.filter_wheel)

    def testSPIRE250(self):
        """ Test SPIRE_250um quantities as example"""

        self.assertTrue(self.filter_wheel["SPIRE_250um"].data_read)
        self.assertEqual(self.filter_wheel["SPIRE_250um"].name, "SPIRE_250um")
        self.assertAlmostEqual(self.filter_wheel["SPIRE_250um"].normfac,
                               3.0796e-3, delta=1e-4)
        self.assertAlmostEqual(self.filter_wheel["SPIRE_250um"].
                               effective_wavelength, 247.268656, delta=1e-4)
        self.assertAlmostEqual(self.filter_wheel["SPIRE_250um"](lambda x: 1),
                               1.011046, delta=1e-4)

    def testAddSpecial(self):
        """ Test adding special members and removing them"""

        self.filter_wheel.add_special("ZSpec_box_1050um_100")
        self.assertTrue("ZSpec_box_1050um_100" in self.filter_wheel)
        self.assertTrue(self.filter_wheel["ZSpec_box_1050um_100"].data_read)
        self.assertEqual(self.filter_wheel["ZSpec_box_1050um_100"].name,
                         "ZSpec_box_1050um_100")
        self.assertAlmostEqual(self.filter_wheel["ZSpec_box_1050um_100"].
                               effective_frequency, 286.1655, delta=1e-3)
        self.assertAlmostEqual(self.filter_wheel["ZSpec_box_1050um_100"]
                               (lambda x: 1), 1.0, delta=1e-4)
        del self.filter_wheel["ZSpec_box_1050um_100"]
        self.assertFalse("ZSpec_box_1050um_100" in self.filter_wheel)


if __name__ == "__main__":
    unittest.main()
