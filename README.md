# Posture Corrector — Reachy Mini + Moondream VLM + AutoResearch

A posture correction system running on Mac Mini (Apple Silicon) that uses a locally hosted **Moondream2** vision-language model to detect posture, drives **Reachy Mini** facial expressions and head movements as corrective feedback, and runs **Karpathy's AutoResearch** overnight to automatically optimize detection prompts and robot behavior parameters.

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
         AutoResearch reads val_score → optimizes config.yaml
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

### 3. Run a timed training session (for AutoResearch)

```bash
python training/train.py --budget 300
```

### 4. Run AutoResearch overnight optimization

Clone Karpathy's AutoResearch, then point it at `training/train.py`:

```bash
git clone https://github.com/karpathy/autoresearch.git ../autoresearch
python ../autoresearch/autoresearch.py \
    --script training/train.py \
    --metric val_score \
    --budget 300
```

AutoResearch will run ~12 experiments/hour, automatically modifying `config.yaml` (prompts, thresholds, timing) and committing improvements to git.

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

AutoResearch treats `training/train.py` as an experiment script and `config.yaml` as the tunable parameter space. Each experiment:

1. LLM reads `training/train.py` + `config.yaml` + past experiment git log
2. Proposes a parameter change (e.g. tighten threshold, rephrase posture prompt)
3. Runs a 5-minute correction session (`--budget 300`)
4. Reads the printed `val_score: X.XX` line
5. Commits to git if score improved; otherwise reverts

### val_score formula

```
val_score = correction_ack_rate − (0.5 × false_positive_rate) + (0.1 × expression_match_rate)
```

- **correction_ack_rate**: fraction of corrections followed by posture improvement within 30s
- **false_positive_rate**: corrections triggered when posture was already good
- **expression_match_rate**: VLM-verified Reachy expressions matched intended expression

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
├── main.py                    # Live runtime loop
├── config.yaml                # Tunable parameters
├── requirements.txt
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
│   ├── train.py                   # AutoResearch target script
│   └── metrics.py                 # val_score computation
└── data/
    └── sessions/                  # Per-session posture logs (gitignored)
```
