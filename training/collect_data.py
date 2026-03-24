"""
One-time labeled dataset collection script.

Run this ONCE while sitting at your desk for ~30 minutes.
It captures frames from your webcam, scores them with Moondream, and saves them
into data/posture_dataset/good/ or data/posture_dataset/bad/ based on the threshold.

After collection, run the overnight optimization loop (local Qwen via mlx-lm):
    python autoresearch_runner.py

Usage:
    python training/collect_data.py [--config config.yaml] [--duration 1800]
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--duration", type=int, default=None,
                        help="Override collect_duration_sec from config")
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_path = Path(config.get("dataset_path", "data/posture_dataset"))
    good_dir = dataset_path / "good"
    bad_dir = dataset_path / "bad"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    interval = config.get("collect_interval_sec", 5)
    duration = args.duration or config.get("collect_duration_sec", 1800)
    threshold = config.get("bad_posture_threshold", 5.0)
    camera_index = config.get("camera_index", 0)

    # Import here so the script fails fast if dependencies are missing
    from src.vision.posture_detector import PostureDetector

    logger.info(f"Loading Moondream model...")
    detector = PostureDetector(config)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Cannot open camera index {camera_index}")
        sys.exit(1)

    logger.info(f"Collecting for {duration}s (interval={interval}s, threshold={threshold})")
    logger.info(f"Dataset: {dataset_path.resolve()}")
    logger.info("Sit naturally — good AND bad posture will be captured. Press Ctrl+C to stop early.\n")

    good_count = 0
    bad_count = 0
    start = time.time()

    try:
        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to capture frame, retrying...")
                time.sleep(1)
                continue

            score, raw = detector.detect(frame)
            ts = time.strftime("%Y%m%d_%H%M%S")

            if score >= threshold:
                dest = good_dir / f"{ts}_{score:.1f}.jpg"
                good_count += 1
            else:
                dest = bad_dir / f"{ts}_{score:.1f}.jpg"
                bad_count += 1

            cv2.imwrite(str(dest), frame)
            elapsed = int(time.time() - start)
            print(f"\r[{elapsed:4d}s/{duration}s] Collected: {good_count} good, {bad_count} bad  (last score={score:.1f})", end="", flush=True)

            time.sleep(interval)

    except KeyboardInterrupt:
        print()
        logger.info("Stopped early by user.")
    finally:
        cap.release()

    print()
    logger.info(f"Done. Dataset saved to {dataset_path.resolve()}")
    logger.info(f"  good frames: {good_count}  ({good_dir})")
    logger.info(f"  bad  frames: {bad_count}  ({bad_dir})")

    if good_count == 0 or bad_count == 0:
        logger.warning("WARNING: One class has 0 frames. Try adjusting your posture during collection, or change bad_posture_threshold in config.yaml.")


if __name__ == "__main__":
    main()
