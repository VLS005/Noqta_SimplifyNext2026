"""Detects irregular walking patterns from a rolling window of sensor readings."""

from collections import deque
from typing import Deque

from config.settings import AgentConfig
from domain.models import GPSPoint, SensorReading, IrregularPattern
from utils.geo import haversine_m


class PatternDetector:
    """Stateful detector - holds a rolling window of recent positions and a
    running wall-hugging streak. One instance per active journey."""

    def __init__(self, config: AgentConfig, baseline_pace_spm: float):
        self.config = config
        self.baseline_pace_spm = baseline_pace_spm
        self.position_history: Deque[GPSPoint] = deque(maxlen=config.position_window)
        self.wall_hug_streak = 0

    def reset(self):
        self.position_history.clear()
        self.wall_hug_streak = 0

    def evaluate(self, reading: SensorReading) -> IrregularPattern:
        """Feed in a new reading and get back the (single, highest-priority)
        irregular pattern detected, or NONE."""
        self.position_history.append(reading.position)
        self._update_wall_hug_streak(reading)

        if self._is_circling():
            return IrregularPattern.CIRCLING
        if self._is_too_slow(reading.steps_per_minute):
            return IrregularPattern.TOO_SLOW
        if self._is_wall_hugging():
            return IrregularPattern.WALL_HUGGING
        return IrregularPattern.NONE

    # -- individual checks ---------------------------------------------------

    def _is_circling(self) -> bool:
        """If we have a full window of positions and the max distance between
        any two points in that window is still small, the user has been
        looping in place rather than making progress."""
        if len(self.position_history) < self.config.position_window:
            return False
        pts = list(self.position_history)
        max_spread = max(
            haversine_m(pts[i], pts[j])
            for i in range(len(pts))
            for j in range(i + 1, len(pts))
        )
        return max_spread < self.config.circling_radius_m

    def _is_too_slow(self, current_pace: float) -> bool:
        if self.baseline_pace_spm <= 0:
            return False  # no personalised baseline yet -> skip this check
        return current_pace < self.baseline_pace_spm * self.config.slow_pace_ratio

    def _update_wall_hug_streak(self, reading: SensorReading):
        if (reading.lateral_wall_distance is not None
                and reading.lateral_wall_distance <= self.config.wall_hug_distance_m):
            self.wall_hug_streak += 1
        else:
            self.wall_hug_streak = 0

    def _is_wall_hugging(self) -> bool:
        return self.wall_hug_streak >= self.config.wall_hug_ticks_required
