# Posture Corrector — Reachy Mini + Moondream VLM + AutoResearch

A posture correction system running on Mac Mini (Apple Silicon) that uses a locally hosted **Moondream2** vision-language model to detect posture, drives **Reachy Mini** facial expressions and head movements as corrective feedback, and runs an **autonomous AutoResearch loop** overnight — powered by a local **Qwen model via mlx-lm** — to automatically optimize detection prompts and thresholds.

---

## Architecture

```
Webcam → Moondream2 VLM → Posture Score
                               │
                    ┌──────────┴──────────┐
                 Bad posture           Good posture
                    │                     │
         Reachy: "concerned"      Reachy: "happy"
         Head: look_at_user       Head: nod
         Speaker: correction TTS
                    │
         VLM checks Reachy's expression (feedback loop)
                    │
         Qwen (mlx-lm) reads val_score → optimizes config.yaml
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the live posture corrector

```bash
python main.py
```

### 3. Build training dataset

**Option A — from YouTube (recommended, no camera needed):**

Downloads public posture education videos, extracts frames, and auto-labels them with Moondream. No personal footage captured.

```bash
python training/collect_from_youtube.py
# or supply your own URLs:
python training/collect_from_youtube.py --urls "https://youtu.be/..." "https://youtu.be/..."
```

Videos are deleted after frame extraction. Frames saved to `data/posture_dataset/` (local only, never committed).

**Option B — from your webcam (personalizes to your exact setup):**

Sit at your desk for ~30 min with natural posture variation. Useful after Option A to add frames that match your specific camera angle and lighting.

```bash
python training/collect_data.py
```

### 4. Run AutoResearch overnight optimization

Starts a local Qwen model (via mlx-lm) that autonomously proposes config changes, evaluates them against your stored dataset, and commits improvements. First run downloads ~4 GB of model weights.

```bash
python autoresearch_runner.py
# or pick a different model:
python autoresearch_runner.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

Runs indefinitely (Ctrl+C to stop). Results logged to `experiments/results.tsv`. No camera, no robot, no internet connection needed after the model downloads.

---

## Configuration (`config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `posture_prompt` | *(VQA string)* | Moondream question for posture scoring |
| `bad_posture_threshold` | `5.0` | Score (1–10) below which correction triggers |
| `correction_cooldown_sec` | `30` | Min seconds between corrections |
| `expression_hold_sec` | `3.0` | Duration of each expression |
| `head_nod_intensity` | `0.15` | Nod movement amplitude (radians) |
| `inference_interval_sec` | `3.0` | Seconds between VLM queries |
| `reachy_host` | `localhost` | Reachy Mini IP/hostname |

---

## AutoResearch Optimization Loop

`autoresearch_runner.py` runs a local **Qwen2.5-7B** model via **mlx-lm** on Apple Silicon. Each experiment:

1. Qwen reads `program.md` (protocol) + `config.yaml` + `experiments/results.tsv` (history)
2. Proposes one config change with a hypothesis (e.g. tighten threshold, rephrase posture prompt)
3. Change is committed to git
4. `training/train.py` runs Moondream against the stored labeled dataset (no camera, no human)
5. Reads the printed `val_score: X.XXXX` line
6. Keeps the commit if score improved; `git revert` if not
7. Loops forever until Ctrl+C

All compute stays on Mac Mini — Qwen on Apple Silicon via MLX, Moondream on MPS.

### val_score formula

```
val_score = 0.6 × sensitivity + 0.4 × specificity
```

- **sensitivity**: fraction of bad-posture frames correctly detected (score < threshold)
- **specificity**: fraction of good-posture frames correctly left alone (score ≥ threshold)

---

## Data Privacy

All data stays on your Mac Mini — nothing is uploaded or committed to git.

| Location | Contents | Stored? |
|---|---|---|
| `data/posture_dataset/` | Labeled JPG frames | Local only (gitignored) |
| `data/youtube_tmp/` | Downloaded videos | Deleted after extraction |
| `data/sessions/` | Live session logs | Local only (gitignored) |
| `experiments/results.tsv` | Metric scores + hypotheses | Git-tracked (no images) |

- YouTube option uses **public** videos only — no personal footage
- Webcam frames from `collect_data.py` are stored locally; delete anytime: `rm -rf data/posture_dataset/`
- All Moondream and Qwen inference runs on-device via MPS/MLX — no cloud API calls

---

## Moondream on Apple Silicon

Moondream2 (1.86B params) runs natively via MPS on M-series Macs:

```python
import moondream as md
model = md.vl(model="moondream-2b")
```

No Ollama server required. Inference runs fully locally.

---

## Project Structure

```
├── main.py                        # Live runtime loop
├── config.yaml                    # Tunable parameters (AutoResearch target)
├── program.md                     # AutoResearch protocol (read by Qwen each iteration)
├── autoresearch_runner.py         # Qwen-powered overnight optimization loop
├── requirements.txt
├── experiments/
│   └── results.tsv                # Experiment history log (git-tracked)
├── src/
│   ├── vision/
│   │   ├── posture_detector.py    # Moondream posture scoring
│   │   └── expression_analyzer.py # VLM expression verification
│   ├── robot/
│   │   ├── reachy_controller.py   # Reachy Mini SDK wrapper
│   │   ├── expressions.py         # Antenna/face expression sequences
│   │   └── movements.py           # Head movement patterns
│   ├── audio/
│   │   └── sound_manager.py       # TTS + WAV playback
│   └── utils/
│       ├── camera.py              # OpenCV frame capture
│       └── logger.py              # Session JSONL logging
├── training/
│   ├── collect_data.py            # One-time labeled dataset capture (user present)
│   ├── train.py                   # Offline evaluator — runs Moondream on stored dataset
│   └── metrics.py                 # val_score / sensitivity / specificity computation
└── data/
    ├── posture_dataset/           # Labeled frames from collect_data.py (gitignored)
    │   ├── good/
    │   └── bad/
    └── sessions/                  # Per-session posture logs (gitignored)
```
