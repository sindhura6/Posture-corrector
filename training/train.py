"""
Offline evaluator — runs Moondream against the stored labeled dataset.

autoresearch_runner.py drives the optimization loop:
  1. Reads program.md + config.yaml + experiments/results.tsv
  2. Asks local Qwen model (via mlx-lm) to propose ONE config change
  3. Commits the change, then runs this script
  4. Reads the printed 'val_score: X.XXXX' line to track improvement
  5. Keeps the commit if score improved; git-reverts otherwise

No live camera, no human presence, no robot required during training.
Run training/collect_data.py once (user present) to build the dataset first.

Usage:
    python training/train.py [--config config.yaml]

AutoResearch loop:
    python autoresearch_runner.py
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import yaml

from src.vision.posture_detector import PostureDetector
from training.metrics import compute_offline_metrics

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_offline_eval(config: dict) -> dict:
    """
    Evaluate config against the stored labeled dataset.
    Returns metrics dict including val_score.
    No camera, no robot, no human required.
    """
    dataset_path = Path(config.get("dataset_path", "data/posture_dataset"))
    good_dir = dataset_path / "good"
    bad_dir = dataset_path / "bad"

    good_frames = sorted(good_dir.glob("*.jpg"))
    bad_frames = sorted(bad_dir.glob("*.jpg"))

    if not good_frames and not bad_frames:
        print("ERROR: No dataset found. Run training/collect_data.py first.", file=sys.stderr)
        print("val_score: 0.0000")
        sys.exit(1)

    threshold = config.get("bad_posture_threshold", 5.0)
    detector = PostureDetector(config)

    # Good frames should score >= threshold (correctly left uncorrected)
    good_correct = 0
    for f in good_frames:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        score, _ = detector.detect(frame)
        if score >= threshold:
            good_correct += 1

    # Bad frames should score < threshold (correctly triggers correction)
    bad_correct = 0
    for f in bad_frames:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        score, _ = detector.detect(frame)
        if score < threshold:
            bad_correct += 1

    return compute_offline_metrics(
        good_correct=good_correct,
        good_total=len(good_frames),
        bad_correct=bad_correct,
        bad_total=len(bad_frames),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_path = Path(config.get("dataset_path", "data/posture_dataset"))
    n_good = len(list((dataset_path / "good").glob("*.jpg")))
    n_bad = len(list((dataset_path / "bad").glob("*.jpg")))
    print(f"Evaluating against dataset: {n_good} good frames, {n_bad} bad frames")

    metrics = run_offline_eval(config)

    print("\n--- Offline Eval Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    # AutoResearch reads this line:
    print(f"\nval_score: {metrics['val_score']:.4f}")


if __name__ == "__main__":
    main()
