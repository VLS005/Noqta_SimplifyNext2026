import pytest

from app.tools.units import mps_to_spm, spm_to_mps


def test_spm_mps_round_trip():
    original = 95.0
    converted = mps_to_spm(spm_to_mps(original, 0.7), 0.7)
    assert converted == pytest.approx(original)


def test_spm_to_mps_known_value():
    # 60 steps/min * 0.7 m = 42 m/min = 0.7 m/s
    assert spm_to_mps(60.0, 0.7) == pytest.approx(0.7)


def test_stride_must_be_positive():
    with pytest.raises(ValueError):
        spm_to_mps(90, 0)
    with pytest.raises(ValueError):
        mps_to_spm(1.0, 0)
