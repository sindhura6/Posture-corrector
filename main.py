"""
Posture Corrector — runtime entry point.

Continuously captures webcam frames, queries Moondream2 for posture score,
and triggers Reachy Mini expressions + head movements accordingly.

Usage:
    python main.py [--config config.yaml]
"""
import argparse
import logging
import signal
import sys
import time
import yaml

from src.utils.camera import Camera
from src.utils.logger import SessionLogger
from src.vision.posture_detector import PostureDetector
from src.vision.expression_analyzer import ExpressionAnalyzer
from src.robot.reachy_controller import ReachyController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_running = True


def _handle_signal(sig, frame):
    global _running
    logger.info("Shutdown signal received.")
    _running = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Reachy Mini Posture Corrector")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    config = load_config(args.config)
    session_id = time.strftime("%Y%m%d_%H%M%S")

    logger.info("Initializing Posture Corrector…")

    # --- Connect to Reachy Mini ---
    reachy = ReachyController(config)
    reachy.connect()

    # --- Load Moondream2 (shared model instance) ---
    detector = PostureDetector(config)
    expr_analyzer = ExpressionAnalyzer(config, detector._model)

    camera_index = config.get("camera_index", 0)
    inference_interval = config.get("inference_interval_sec", 3.0)
    bad_threshold = config.get("bad_posture_threshold", 5.0)
    cooldown = config.get("correction_cooldown_sec", 30.0)

    last_correction_ts = 0.0
    last_expression = "neutral"

    logger.info("Starting posture correction loop. Press Ctrl+C to stop.")

    with Camera(index=camera_index) as cam, SessionLogger(session_id) as slog:
        while _running:
            frame = cam.capture()
            if frame is None:
                time.sleep(0.5)
                continue

            # --- Posture detection ---
            score, raw = detector.detect(frame)
            slog.log_score(score, raw)
            logger.info(f"Posture score: {score:.1f}  ('{raw}')")

            now = time.time()
            if score < bad_threshold:
                # Bad posture
                if (now - last_correction_ts) >= cooldown:
                    reachy.react_bad_posture(score)
                    last_correction_ts = now
                    last_expression = "concerned"
                    slog.log_correction(score, "bad_posture_reaction")

                    # --- Expression verification via VLM ---
                    if config.get("expression_check_enabled", True):
                        reachy_frame = cam.capture()
                        if reachy_frame is not None:
                            detected_expr = expr_analyzer.analyze(reachy_frame)
                            slog.log_expression(last_expression, detected_expr)
                            if detected_expr != last_expression:
                                logger.info(
                                    f"Expression mismatch: expected '{last_expression}', "
                                    f"detected '{detected_expr}'"
                                )
            else:
                # Good posture
                reachy.react_good_posture(score)
                last_expression = "happy"

            time.sleep(inference_interval)

    reachy.disconnect()
    logger.info(f"Session complete. Log saved to data/sessions/{session_id}.jsonl")


if __name__ == "__main__":
    main()
