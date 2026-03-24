"""
Audio wrapper for Reachy Mini.
Supports text-to-speech and WAV sound file playback via the robot's speaker.
"""
import logging
import os

logger = logging.getLogger(__name__)


class SoundManager:
    def __init__(self, robot):
        self._robot = robot

    def say(self, text: str):
        """Speak text via Reachy's onboard TTS."""
        try:
            self._robot.speaker.say(text)
        except Exception as e:
            logger.warning(f"TTS error: {e}")
            print(f"[AUDIO] {text}")

    def play_sound(self, filepath: str):
        """Play a WAV file from disk via Reachy's speaker."""
        if not os.path.isfile(filepath):
            logger.warning(f"Sound file not found: {filepath}")
            return
        try:
            self._robot.speaker.play_sound(filepath)
        except Exception as e:
            logger.warning(f"Sound playback error ({filepath}): {e}")
