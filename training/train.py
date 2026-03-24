"""
AutoResearch training target script.
AutoResearch (github.com/karpathy/autoresearch) will:
  1. Read this script + config.yaml
  2. Propose config changes via LLM
  3. Run this script for BUDGET_SECS seconds
  4. Read the printed 'val_score: X.XX' line to track improvement
  5. Commit improvements to git if the score improves

Usage:
    python training/train.py [--budget 300]

AutoResearch call:
    python autoresearch.py --script training/train.py --metric val_score --budget 300
"""
import argparse
import logging
import sys
import time
import yaml

from src.utils.camera import Camera
from src.utils.logger import SessionLogger
from src.vision.posture_detector import PostureDetector
from src.robot.reachy_controller import ReachyController
from training.metrics import compute_metrics

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_session(config: dict, budget_secs: float, session_id: str) -> str:
    """Run a timed posture correction session. Returns the session log path."""
    camera_index = config.get("camera_index", 0)
    inference_interval = config.get("inference_interval_sec", 3.0)
    bad_threshold = config.get("bad_posture_threshold", 5.0)
    cooldown = config.get("correction_cooldown_sec", 30.0)

    reachy = ReachyController(config)
    reachy.connect()

    detector = PostureDetector(config)

    log_path = f"data/sessions/{session_id}.jsonl"

    with Camera(index=camera_index) as cam, SessionLogger(session_id) as slog:
        last_correction_ts = 0.0
        session_start = time.time()

        while time.time() - session_start < budget_secs:
            frame = cam.capture()
            if frame is None:
                time.sleep(0.5)
                continue

            score, raw = detector.detect(frame)
            slog.log_score(score, raw)

            now = time.time()
            if score < bad_threshold and (now - last_correction_ts) >= cooldown:
                import random
                msg = random.choice(config.get("tts_bad_posture_messages", ["Sit up straight!"]))
                reachy.react_bad_posture(score)
                slog.log_correction(score, msg)
                last_correction_ts = now
            elif score >= bad_threshold:
                reachy.react_good_posture(score)

            time.sleep(inference_interval)

    reachy.disconnect()
    return log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=300,
                        help="Experiment duration in seconds (default: 300)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    session_id = time.strftime("%Y%m%d_%H%M%S")

    print(f"Starting {args.budget}s posture correction session ({session_id})")
    log_path = run_session(config, float(args.budget), session_id)

    metrics = compute_metrics(log_path)
    print(f"\n--- Session Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    # AutoResearch reads this line:
    print(f"\nval_score: {metrics['val_score']:.4f}")


if __name__ == "__main__":
    main()
