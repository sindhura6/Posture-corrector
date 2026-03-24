"""
Named facial expression sequences for Reachy Mini.
Each expression is a sequence of antenna + face display states.
Reachy Mini expresses emotion primarily through its antenna LEDs and face display.
"""
import logging
import time

logger = logging.getLogger(__name__)


class ExpressionPlayer:
    """Plays named expression animations on Reachy Mini."""

    EXPRESSIONS = {
        "happy": [
            {"antenna_left": 1.0, "antenna_right": 1.0, "hold": 0.3},
            {"antenna_left": 0.6, "antenna_right": 0.6, "hold": 0.2},
            {"antenna_left": 1.0, "antenna_right": 1.0, "hold": 0.3},
        ],
        "concerned": [
            {"antenna_left": -0.4, "antenna_right": 0.4, "hold": 0.5},
            {"antenna_left": 0.0, "antenna_right": 0.0, "hold": 0.3},
            {"antenna_left": -0.4, "antenna_right": 0.4, "hold": 0.5},
        ],
        "sad": [
            {"antenna_left": -0.8, "antenna_right": -0.8, "hold": 0.8},
            {"antenna_left": -0.5, "antenna_right": -0.5, "hold": 0.4},
        ],
        "neutral": [
            {"antenna_left": 0.0, "antenna_right": 0.0, "hold": 0.5},
        ],
    }

    def __init__(self, robot):
        self._robot = robot

    def play(self, name: str, hold_sec: float = 3.0):
        """Play a named expression, then hold the final state for hold_sec."""
        sequence = self.EXPRESSIONS.get(name, self.EXPRESSIONS["neutral"])
        try:
            for frame in sequence:
                self._set_antennas(frame["antenna_left"], frame["antenna_right"])
                time.sleep(frame["hold"])
            time.sleep(max(0.0, hold_sec - sum(f["hold"] for f in sequence)))
            self._reset_antennas()
        except Exception as e:
            logger.warning(f"Expression '{name}' failed: {e}")

    def _set_antennas(self, left: float, right: float):
        try:
            self._robot.left_antenna.goal_position = left
            self._robot.right_antenna.goal_position = right
        except AttributeError:
            logger.debug("Antenna control not available on this Reachy model.")

    def _reset_antennas(self):
        self._set_antennas(0.0, 0.0)
