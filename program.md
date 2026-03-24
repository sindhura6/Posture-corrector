# Posture Corrector: AutoResearch Experiment Protocol

This file is read by `autoresearch_runner.py` each iteration and included in the prompt to the local Qwen model. It defines the rules of the autonomous optimization loop.

## Objective

Maximize `val_score = 0.6 × sensitivity + 0.4 × specificity`.

- **sensitivity**: fraction of bad-posture frames correctly detected (score < threshold)
- **specificity**: fraction of good-posture frames correctly left alone (score >= threshold)

Higher val_score = better posture detection. Target: sensitivity ≥ 0.85, specificity ≥ 0.80.

## Mutable (you may change these config.yaml keys)

- `posture_prompt` — the yes/no question Moondream answers about the user's posture
- `bad_posture_threshold` — score (1–10) below which correction triggers (range: 3.0–8.0)
- `inference_interval_sec` — seconds between VLM queries (do not touch unless justified)

## Fixed (do NOT modify)

- `training/train.py` — evaluation harness, locked
- `training/metrics.py` — metric computation, locked
- `src/vision/posture_detector.py` — VLM inference, locked

## Response Format (mandatory — runner parses these exact markers)

```
HYPOTHESIS: <one sentence explaining why this change should improve val_score>
CONFIG_YAML:
```yaml
<complete config.yaml content with exactly ONE change applied>
```
```

## Experiment Guidelines

- Make exactly one change per experiment so results are interpretable
- `posture_prompt` must be a question answerable as yes/no by a VLM given a webcam image
- Prompt changes: try specificity (spine angle, shoulder level, head tilt) over vague terms
- Threshold changes: lower threshold = more corrections (higher sensitivity, lower specificity)
- Do not add or remove config.yaml keys — only change values of existing keys
- Prefer simple, testable hypotheses over complex multi-variable changes
