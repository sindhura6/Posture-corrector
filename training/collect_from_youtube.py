"""
Build the posture dataset from YouTube videos — no user presence required.

Downloads public posture education videos using yt-dlp, extracts frames with
OpenCV, and auto-labels each frame with Moondream. Saves labeled frames to
data/posture_dataset/good/ and data/posture_dataset/bad/.

All data stays local. Videos are deleted after frame extraction by default
(pass --keep-videos to retain them). No personal footage is captured or stored.

Usage:
    python training/collect_from_youtube.py
    python training/collect_from_youtube.py --urls "https://youtu.be/..." "https://youtu.be/..."
    python training/collect_from_youtube.py --keep-videos
    python training/collect_from_youtube.py --interval 3 --config config.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Curated public posture education videos — office/desk posture demos
# showing clear good vs bad sitting posture from various angles.
DEFAULT_YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=kx0C6V_Zn8s",  # Posture Perfect: Good vs Bad posture
    "https://www.youtube.com/watch?v=OyK0oE5rwFY",  # How to sit correctly at a desk
    "https://www.youtube.com/watch?v=3OzHtLEzDzk",  # Bad posture habits (office)
    "https://www.youtube.com/watch?v=qO3_SVSmJuE",  # Ergonomic sitting posture guide
    "https://www.youtube.com/watch?v=RqcOCBb4arc",  # Forward head posture correction
]


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_video(url: str, output_dir: Path) -> Path | None:
    """Download a YouTube video to output_dir using yt-dlp. Returns path to video file."""
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError:
        logger.error("yt-dlp not installed. Run: pip install yt-dlp")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "unknown")
            video_path = output_dir / f"{video_id}.mp4"
            if video_path.exists():
                return video_path
            # yt-dlp may use a different extension
            for f in output_dir.glob(f"{video_id}.*"):
                if f.suffix in (".mp4", ".webm", ".mkv"):
                    return f
        logger.warning(f"Downloaded but could not locate file for {url}")
        return None
    except Exception as exc:
        logger.warning(f"Failed to download {url}: {exc}")
        return None


def extract_and_label_frames(
    video_path: Path,
    detector,
    good_dir: Path,
    bad_dir: Path,
    threshold: float,
    interval_sec: float,
    video_index: int,
    total_videos: int,
) -> tuple[int, int]:
    """Extract frames from video, label with Moondream, save to dataset dirs.

    Returns (good_count, bad_count).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(fps * interval_sec))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    good_count = 0
    bad_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            score, _ = detector.detect(frame)
            label = "good" if score >= threshold else "bad"
            dest_dir = good_dir if label == "good" else bad_dir
            dest = dest_dir / f"yt{video_index:02d}_{frame_idx:06d}_{score:.1f}.jpg"
            cv2.imwrite(str(dest), frame)
            if label == "good":
                good_count += 1
            else:
                bad_count += 1

            elapsed_sec = int(frame_idx / fps)
            print(
                f"\r  [video {video_index}/{total_videos}]"
                f"  {elapsed_sec}s / {int(total_frames/fps)}s"
                f"  → {label} (score={score:.1f})"
                f"  total: {good_count} good, {bad_count} bad",
                end="",
                flush=True,
            )

        frame_idx += 1

    cap.release()
    print()
    return good_count, bad_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build posture dataset from YouTube videos (no camera required)"
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        metavar="URL",
        help="YouTube URLs to process (default: built-in curated list)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file path (default: config.yaml)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds between extracted frames (overrides config youtube_frame_interval_sec)",
    )
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Keep downloaded video files after frame extraction (default: delete them)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    urls = args.urls or DEFAULT_YOUTUBE_URLS
    interval = args.interval or config.get("youtube_frame_interval_sec", 2.0)
    threshold = config.get("bad_posture_threshold", 5.0)
    dataset_path = Path(config.get("dataset_path", "data/posture_dataset"))
    tmp_dir = Path(config.get("youtube_tmp_dir", "data/youtube_tmp"))

    good_dir = dataset_path / "good"
    bad_dir = dataset_path / "bad"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Moondream model...")
    from src.vision.posture_detector import PostureDetector  # noqa: PLC0415
    detector = PostureDetector(config)

    logger.info(f"Processing {len(urls)} video(s)  (frame every {interval}s, threshold={threshold})")
    logger.info(f"Dataset: {dataset_path.resolve()}")
    logger.info("All data stays local. Videos deleted after extraction unless --keep-videos.\n")

    total_good = 0
    total_bad = 0

    for i, url in enumerate(urls, start=1):
        logger.info(f"[{i}/{len(urls)}] Downloading: {url}")
        video_path = download_video(url, tmp_dir)
        if video_path is None:
            logger.warning(f"Skipping {url}")
            continue

        logger.info(f"  Extracting frames from {video_path.name}...")
        good, bad = extract_and_label_frames(
            video_path=video_path,
            detector=detector,
            good_dir=good_dir,
            bad_dir=bad_dir,
            threshold=threshold,
            interval_sec=interval,
            video_index=i,
            total_videos=len(urls),
        )
        total_good += good
        total_bad += bad
        logger.info(f"  Done: {good} good, {bad} bad frames")

        if not args.keep_videos:
            video_path.unlink(missing_ok=True)
            logger.info(f"  Deleted {video_path.name}")

    # Clean up tmp dir if empty
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()

    print()
    logger.info(f"Dataset complete: {total_good} good frames, {total_bad} bad frames")
    logger.info(f"  Saved to: {dataset_path.resolve()}")
    logger.info(f"  To run AutoResearch: python autoresearch_runner.py")

    if total_good == 0 or total_bad == 0:
        logger.warning(
            "WARNING: one class has 0 frames. "
            "Try adjusting bad_posture_threshold in config.yaml, or add more videos."
        )


if __name__ == "__main__":
    main()
