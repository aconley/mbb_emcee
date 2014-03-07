""" Unit tests for response"""
from mbb_emcee import response, response_set
from numpy.testing import assert_allclose


def setup_module():
    global __WHEEL__
    __WHEEL__ = response_set()


def test_DefaultMember():
    """ Test whether default set has expected members"""

    assert "SCUBA2_850um" in __WHEEL__
    assert "SPIRE_250um" in __WHEEL__
    assert "SPIRE_350um" in __WHEEL__
    assert "SPIRE_500um" in __WHEEL__
    assert "Bolocam_1.1mm" in __WHEEL__


def test_SPIRE250():
    """ Test SPIRE_250um quantities as example"""

    assert __WHEEL__["SPIRE_250um"].data_read
    assert __WHEEL__["SPIRE_250um"].name == "SPIRE_250um"
    assert_allclose(__WHEEL__["SPIRE_250um"].normfac, 3.0796e-3, atol=1e-4)
    assert_allclose(__WHEEL__["SPIRE_250um"].effective_wavelength,
                    247.268656, atol=1e-4)
    assert_allclose(__WHEEL__["SPIRE_250um"](lambda x: 1),
                    1.011046, atol=1e-4)


def testAddSpecial():
    """ Test adding special members and removing them"""

    __WHEEL__.add_special("ZSpec_box_1050um_100")
    assert "ZSpec_box_1050um_100" in __WHEEL__
    assert __WHEEL__["ZSpec_box_1050um_100"].data_read
    assert __WHEEL__["ZSpec_box_1050um_100"].name == "ZSpec_box_1050um_100"
    assert_allclose(__WHEEL__["ZSpec_box_1050um_100"].
                    effective_frequency, 286.1655, atol=1e-3)
    assert_allclose(__WHEEL__["ZSpec_box_1050um_100"]
                    (lambda x: 1), 1.0, atol=1e-4)
    del __WHEEL__["ZSpec_box_1050um_100"]
    assert not "ZSpec_box_1050um_100" in __WHEEL__
