"""
Session logging for posture scores and correction events.
Writes newline-delimited JSON to data/sessions/.
"""
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SESSION_DIR = Path("data/sessions")


class SessionLogger:
    def __init__(self, session_id: str | None = None):
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        if session_id is None:
            session_id = time.strftime("%Y%m%d_%H%M%S")
        self._path = SESSION_DIR / f"{session_id}.jsonl"
        self._file = open(self._path, "a")
        logger.info(f"Session log: {self._path}")

    def log_score(self, score: float, raw: str = ""):
        self._write({"type": "score", "score": score, "raw": raw, "ts": time.time()})

    def log_correction(self, score: float, message: str):
        self._write({"type": "correction", "score": score, "message": message, "ts": time.time()})

    def log_expression(self, expected: str, detected: str):
        self._write({"type": "expression_check", "expected": expected,
                     "detected": detected, "match": expected == detected, "ts": time.time()})

    def _write(self, record: dict):
        try:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()
        except Exception as e:
            logger.warning(f"Log write failed: {e}")

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load_session(session_id: str) -> list[dict]:
    path = SESSION_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
