"""
OpenCV webcam capture utility.
Provides simple frame-grab with error handling and optional resize.
"""
import logging
import cv2

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self._index = index
        self._width = width
        self._height = height
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self._index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        logger.info(f"Camera {self._index} opened ({self._width}x{self._height})")

    def capture(self):
        """Return a BGR frame or None on failure."""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Failed to capture frame.")
            return None
        return frame

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None
            logger.info("Camera closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
