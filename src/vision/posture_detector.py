"""
Moondream2-based posture detector.
Sends webcam frames to the local Moondream VLM and parses a 1–10 posture score.
"""
import re
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class PostureDetector:
    def __init__(self, config: dict):
        self.prompt = config["posture_prompt"]
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            import moondream as md
            self._model = md.vl(model="moondream-2b")
            logger.info("Moondream2 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Moondream model: {e}")
            raise

    def detect(self, frame_bgr) -> tuple[float, str]:
        """
        Args:
            frame_bgr: OpenCV BGR numpy array from the webcam.
        Returns:
            (score, raw_text) where score is 1.0–10.0.
            Returns (5.0, raw_text) on parse failure.
        """
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        try:
            encoded = self._model.encode_image(pil_image)
            result = self._model.query(encoded, self.prompt)
            raw = result.get("answer", "").strip()
            score = self._parse_score(raw)
            logger.debug(f"Posture score: {score:.1f}  raw='{raw}'")
            return score, raw
        except Exception as e:
            logger.warning(f"Posture detection error: {e}")
            return 5.0, ""

    def _parse_score(self, text: str) -> float:
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if match:
            score = float(match.group(1))
            return max(1.0, min(10.0, score))
        return 5.0
