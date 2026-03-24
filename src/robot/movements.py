"""
Head movement patterns for Reachy Mini.
Uses the head's roll/pitch/yaw joints to draw the user's attention and
signal acknowledgment.
"""
import logging
import time

logger = logging.getLogger(__name__)


class MovementPlayer:
    """Executes named head movement sequences on Reachy Mini."""

    def __init__(self, robot, config: dict):
        self._robot = robot
        self._nod_intensity = config.get("head_nod_intensity", 0.15)
        self._look_intensity = config.get("head_look_intensity", 0.2)

    def look_at_user(self):
        """Tilt head slightly toward the camera to draw attention."""
        self._move_head(pitch=self._look_intensity, duration=0.5)
        time.sleep(0.8)
        self._move_head(pitch=0.0, duration=0.4)

    def nod(self):
        """Two-beat positive nod."""
        for _ in range(2):
            self._move_head(pitch=self._nod_intensity, duration=0.25)
            time.sleep(0.3)
            self._move_head(pitch=-self._nod_intensity * 0.5, duration=0.2)
            time.sleep(0.25)
        self._move_head(pitch=0.0, duration=0.3)

    def head_tilt_concern(self):
        """Slight roll tilt expressing concern."""
        self._move_head(roll=self._look_intensity * 0.5, duration=0.5)
        time.sleep(0.6)
        self._move_head(roll=0.0, duration=0.4)

    def _move_head(self, pitch: float = 0.0, roll: float = 0.0, yaw: float = 0.0,
                   duration: float = 0.3):
        try:
            head = self._robot.head
            head.neck.roll.goal_position = roll
            head.neck.pitch.goal_position = pitch
            head.neck.yaw.goal_position = yaw
            time.sleep(duration)
        except AttributeError as e:
            logger.debug(f"Head control not available: {e}")
        except Exception as e:
            logger.warning(f"Head movement error: {e}")
