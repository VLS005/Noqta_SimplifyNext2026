"""Walking-pace unit conversion between steps/min and metres/second.

This uses a fixed stride length as an approximation, not a personalised
measurement. A later version could derive stride from GPS-distance / step
count per journey if those are tracked together.
"""

from __future__ import annotations


def spm_to_mps(pace_spm: float, stride_length_m: float = 0.7) -> float:
    """steps/min * stride_m = metres/min; / 60 = metres/sec."""
    if stride_length_m <= 0:
        raise ValueError("stride_length_m must be positive")
    return (pace_spm * stride_length_m) / 60.0


def mps_to_spm(pace_mps: float, stride_length_m: float = 0.7) -> float:
    """Inverse of spm_to_mps."""
    if stride_length_m <= 0:
        raise ValueError("stride_length_m must be positive")
    return (pace_mps * 60.0) / stride_length_m
