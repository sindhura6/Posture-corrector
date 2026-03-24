"""
Evaluation metrics for AutoResearch experiment runs.

Two evaluation modes:
  - compute_offline_metrics(): used by train.py — evaluates against stored dataset
    (no camera/human needed, used for overnight AutoResearch optimization)
  - compute_metrics(): used by main.py — reads live session JSONL logs
    (requires user presence, useful for monitoring real sessions)
"""
import json
from pathlib import Path


def compute_offline_metrics(
    good_correct: int,
    good_total: int,
    bad_correct: int,
    bad_total: int,
) -> dict:
    """
    Compute offline classification metrics from dataset evaluation.
    Used by training/train.py (AutoResearch target — no camera/human required).

    Args:
        good_correct: frames in good/ that scored >= threshold (true negatives)
        good_total:   total frames in good/
        bad_correct:  frames in bad/ that scored < threshold (true positives)
        bad_total:    total frames in bad/

    Returns:
        dict with sensitivity, specificity, val_score (higher is better).
    """
    sensitivity = bad_correct / bad_total if bad_total > 0 else 0.0   # recall on bad posture
    specificity = good_correct / good_total if good_total > 0 else 0.0  # recall on good posture

    # Weighted: catching bad posture (sensitivity) matters more than avoiding false alarms
    val_score = 0.6 * sensitivity + 0.4 * specificity

    return {
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "bad_correct": bad_correct,
        "bad_total": bad_total,
        "good_correct": good_correct,
        "good_total": good_total,
        "val_score": round(val_score, 4),
    }


def compute_metrics(session_path: str, ack_window_sec: float = 30.0) -> dict:
    """
    Compute posture correction metrics from a session log.

    Args:
        session_path: Path to a .jsonl session file.
        ack_window_sec: Seconds after a correction to look for posture improvement.

    Returns:
        dict with:
            correction_ack_rate   – fraction of corrections followed by improvement
            false_positive_rate   – corrections fired when posture was already good (score ≥ 7)
            expression_match_rate – fraction of expression checks that matched expected
            val_score             – composite metric (higher is better)
    """
    path = Path(session_path)
    if not path.exists():
        return {"val_score": 0.0}

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    scores = [r for r in records if r["type"] == "score"]
    corrections = [r for r in records if r["type"] == "correction"]
    expr_checks = [r for r in records if r["type"] == "expression_check"]

    # --- Correction acknowledgment rate ---
    ack_count = 0
    for corr in corrections:
        corr_ts = corr["ts"]
        # Find the next score recorded after this correction within the window
        for s in scores:
            if corr_ts < s["ts"] <= corr_ts + ack_window_sec:
                if s["score"] >= corr["score"] + 1.0:  # meaningful improvement
                    ack_count += 1
                break

    ack_rate = ack_count / len(corrections) if corrections else 0.0

    # --- False positive rate ---
    good_threshold = 7.0
    false_positives = sum(1 for c in corrections if c.get("score", 0) >= good_threshold)
    fp_rate = false_positives / len(corrections) if corrections else 0.0

    # --- Expression match rate ---
    expr_matches = sum(1 for e in expr_checks if e.get("match", False))
    expr_match_rate = expr_matches / len(expr_checks) if expr_checks else 1.0

    # --- Composite val_score (maximise) ---
    val_score = ack_rate - (0.5 * fp_rate) + (0.1 * expr_match_rate)

    return {
        "correction_ack_rate": round(ack_rate, 4),
        "false_positive_rate": round(fp_rate, 4),
        "expression_match_rate": round(expr_match_rate, 4),
        "n_corrections": len(corrections),
        "n_scores": len(scores),
        "val_score": round(val_score, 4),
    }
