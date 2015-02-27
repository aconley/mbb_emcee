""" Unit tests for response"""
import pytest
from mbb_emcee import response_set
from numpy.testing import assert_allclose

@pytest.fixture(scope="module")
def wheel():
    return response_set()
        
def test_default_member(wheel):
    """ Test whether default set has expected members"""

    assert "SCUBA2_850um" in wheel
    assert "SPIRE_250um" in wheel
    assert "SPIRE_350um" in wheel
    assert "SPIRE_500um" in wheel
    assert "Bolocam_1.1mm" in wheel

def test_spire250(wheel):
    """ Test SPIRE_250um quantities as example"""

    assert wheel["SPIRE_250um"].data_read
    assert wheel["SPIRE_250um"].name == "SPIRE_250um"
    assert_allclose(wheel["SPIRE_250um"].normfac, 3.0796e-3,
                    atol=1e-4)
    assert_allclose(wheel["SPIRE_250um"].effective_wavelength,
                    247.268656, atol=1e-4)
    assert_allclose(wheel["SPIRE_250um"](lambda x: 1),
                    1.011046, atol=1e-4)

def test_add_special(wheel):
    """ Test adding special members and removing them"""

    wheel.add_special("ZSpec_box_1050um_100")
    assert "ZSpec_box_1050um_100" in wheel
    assert wheel["ZSpec_box_1050um_100"].data_read
    assert wheel["ZSpec_box_1050um_100"].name == "ZSpec_box_1050um_100"
    assert_allclose(wheel["ZSpec_box_1050um_100"].
                    effective_frequency, 286.1655, atol=1e-3)
    assert_allclose(wheel["ZSpec_box_1050um_100"]
                    (lambda x: 1), 1.0, atol=1e-4)
    del wheel["ZSpec_box_1050um_100"]
    assert not "ZSpec_box_1050um_100" in wheel
    
