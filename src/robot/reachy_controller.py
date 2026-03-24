"""
High-level wrapper around the Reachy Mini SDK.
Handles connection lifecycle, expression, movement, and audio dispatch.
"""
import logging
import random

from src.robot.expressions import ExpressionPlayer
from src.robot.movements import MovementPlayer
from src.audio.sound_manager import SoundManager

logger = logging.getLogger(__name__)


class ReachyController:
    def __init__(self, config: dict):
        self._config = config
        self._robot = None
        self._expressions = None
        self._movements = None
        self._sound = None
        self._bad_messages = config.get("tts_bad_posture_messages", ["Please sit up straight!"])
        self._good_message = config.get("tts_good_posture_message", "Great posture!")

    def connect(self):
        """Connect to Reachy Mini and initialize sub-systems."""
        host = self._config.get("reachy_host", "localhost")
        try:
            from reachy_mini import ReachyMini
            self._robot = ReachyMini(host=host)
            logger.info(f"Connected to Reachy Mini at {host}")
        except ImportError:
            logger.warning("reachy-mini-sdk not installed. Running in mock mode.")
            self._robot = _MockRobot()
        except Exception as e:
            logger.error(f"Could not connect to Reachy Mini at {host}: {e}")
            logger.warning("Falling back to mock robot.")
            self._robot = _MockRobot()

        self._expressions = ExpressionPlayer(self._robot)
        self._movements = MovementPlayer(self._robot, self._config)
        self._sound = SoundManager(self._robot)

    def disconnect(self):
        try:
            if self._robot and not isinstance(self._robot, _MockRobot):
                self._robot.turn_off()
        except Exception as e:
            logger.warning(f"Disconnect error: {e}")

    # --- High-level posture feedback actions ---

    def react_bad_posture(self, score: float):
        """Trigger concerned expression, attention movement, and correction audio."""
        hold = self._config.get("expression_hold_sec", 3.0)
        self._expressions.play("concerned", hold_sec=hold)
        self._movements.look_at_user()
        msg = random.choice(self._bad_messages)
        self._sound.say(msg)
        logger.info(f"Bad posture reaction (score={score:.1f}): '{msg}'")

    def react_good_posture(self, score: float):
        """Trigger happy expression and acknowledgment nod."""
        hold = self._config.get("expression_hold_sec", 3.0)
        self._expressions.play("happy", hold_sec=hold)
        self._movements.nod()
        logger.info(f"Good posture acknowledged (score={score:.1f})")

    def show_expression(self, name: str):
        hold = self._config.get("expression_hold_sec", 3.0)
        self._expressions.play(name, hold_sec=hold)

    def do_movement(self, name: str):
        fn = getattr(self._movements, name, None)
        if fn:
            fn()
        else:
            logger.warning(f"Unknown movement: {name}")

    def speak(self, text: str):
        self._sound.say(text)

    def get_robot(self):
        return self._robot


class _MockRobot:
    """Stub robot used when Reachy Mini hardware is unavailable."""

    class _Speaker:
        def say(self, text):
            print(f"[MOCK SPEAKER] {text}")

        def play_sound(self, path):
            print(f"[MOCK SPEAKER] play: {path}")

    class _Antenna:
        goal_position = 0.0

    class _Neck:
        class _Joint:
            goal_position = 0.0
        roll = _Joint()
        pitch = _Joint()
        yaw = _Joint()

    class _Head:
        neck = None

    def __init__(self):
        self.speaker = self._Speaker()
        self.left_antenna = self._Antenna()
        self.right_antenna = self._Antenna()
        head = self._Head()
        head.neck = self._Neck()
        self.head = head

    def turn_off(self):
        print("[MOCK ROBOT] turned off")
