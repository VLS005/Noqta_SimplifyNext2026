"""Run with: python -m pytest tests/  (or python tests/test_detection.py)"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import AgentConfig
from domain.models import GPSPoint, SensorReading, IrregularPattern
from domain.detection import PatternDetector


def test_too_slow_detected():
    config = AgentConfig(position_window=3)
    detector = PatternDetector(config, baseline_pace_spm=100)
    reading = SensorReading(
        position=GPSPoint(1.3, 103.8, 0),
        heading_deg=0,
        steps_per_minute=30,  # well below 50% of baseline
    )
    assert detector.evaluate(reading) == IrregularPattern.TOO_SLOW


def test_normal_pace_not_flagged():
    config = AgentConfig(position_window=3)
    detector = PatternDetector(config, baseline_pace_spm=100)
    reading = SensorReading(
        position=GPSPoint(1.3, 103.8, 0),
        heading_deg=0,
        steps_per_minute=95,
    )
    assert detector.evaluate(reading) == IrregularPattern.NONE


def test_circling_detected():
    config = AgentConfig(position_window=4, circling_radius_m=15.0)
    detector = PatternDetector(config, baseline_pace_spm=0)  # disable pace check
    base_lat, base_lon = 1.3000, 103.8000
    result = IrregularPattern.NONE
    for i in range(4):
        reading = SensorReading(
            position=GPSPoint(base_lat + 0.000001 * i, base_lon, i),
            heading_deg=0,
            steps_per_minute=90,
        )
        result = detector.evaluate(reading)
    assert result == IrregularPattern.CIRCLING


def test_wall_hugging_detected():
    config = AgentConfig(wall_hug_ticks_required=3, wall_hug_distance_m=0.3)
    detector = PatternDetector(config, baseline_pace_spm=0)
    result = IrregularPattern.NONE
    for i in range(3):
        reading = SensorReading(
            position=GPSPoint(1.3 + 0.001 * i, 103.8, i),  # far apart -> not circling
            heading_deg=0,
            steps_per_minute=90,
            lateral_wall_distance=0.1,
        )
        result = detector.evaluate(reading)
    assert result == IrregularPattern.WALL_HUGGING


if __name__ == "__main__":
    test_too_slow_detected()
    test_normal_pace_not_flagged()
    test_circling_detected()
    test_wall_hugging_detected()
    print("All tests passed.")
