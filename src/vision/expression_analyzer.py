"""
Uses Moondream to inspect Reachy Mini's onboard camera and verify
what expression the robot is currently displaying. Used as a reward
signal in the AutoResearch optimization loop.
"""
import logging
from PIL import Image

logger = logging.getLogger(__name__)

VALID_EXPRESSIONS = {"happy", "sad", "concerned", "neutral"}


class ExpressionAnalyzer:
    def __init__(self, config: dict, model):
        """
        Args:
            config: loaded config dict
            model: already-loaded Moondream model instance (shared with PostureDetector)
        """
        self.prompt = config.get("expression_prompt", "")
        self.enabled = config.get("expression_check_enabled", True)
        self._model = model

    def analyze(self, frame_bgr) -> str:
        """
        Capture Reachy's expression from a frame and return a label.

        Returns one of: 'happy', 'sad', 'concerned', 'neutral', or 'unknown'
        """
        if not self.enabled or self._model is None:
            return "unknown"

        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        try:
            encoded = self._model.encode_image(pil_image)
            result = self._model.query(encoded, self.prompt)
            raw = result.get("answer", "").strip().lower()
            for label in VALID_EXPRESSIONS:
                if label in raw:
                    logger.debug(f"Detected Reachy expression: {label}")
                    return label
            logger.debug(f"Expression unrecognized, raw='{raw}'")
            return "unknown"
        except Exception as e:
            logger.warning(f"Expression analysis error: {e}")
            return "unknown"
