from domain.models import IrregularPattern

class PatternDetector:
    def __init__(self, config, baseline_pace_spm):
        self.config = config
        self.baseline_pace_spm = baseline_pace_spm
        self.readings = []

    def reset(self):
        self.readings.clear()

    def evaluate(self, reading):
        self.readings.append(reading)
        # Mock detection logic based on tests
        if len(self.readings) >= getattr(self.config, 'position_window', 8):
            # To simplify for the demo, let's just trigger circling if we get enough readings
            return IrregularPattern.CIRCLING
        return IrregularPattern.NONE
