"""
AutoResearch runner — local Qwen model drives the optimization loop on Apple Silicon.

Uses mlx-lm to run Qwen locally (no external API calls). Each iteration:
  1. Reads program.md + config.yaml + experiments/results.tsv
  2. Asks Qwen to propose ONE config.yaml change with a hypothesis
  3. Commits the change, runs training/train.py
  4. Reads val_score from stdout
  5. Keeps the change if improved; reverts via git if not
  6. Logs to experiments/results.tsv
  7. Loops indefinitely (Ctrl+C to stop)

Usage:
    python autoresearch_runner.py
    python autoresearch_runner.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
    python autoresearch_runner.py --max-experiments 50

First run downloads the Qwen model (~4 GB, cached in ~/.cache/huggingface/).
"""
import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
RESULTS_FILE = Path("experiments/results.tsv")
TRAIN_CMD = [sys.executable, "training/train.py"]


def read_file(path: str) -> str:
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return f"(file not found: {path})"


def build_prompt(program_md: str, config_yaml: str, results_history: str) -> str:
    return (
        "You are an autonomous research agent optimizing posture detection parameters.\n\n"
        "## Protocol\n"
        f"{program_md}\n\n"
        "## Current config.yaml\n"
        f"```yaml\n{config_yaml}\n```\n\n"
        "## Experiment history (experiments/results.tsv)\n"
        f"{results_history or '(no experiments yet — this is experiment #1)'}\n\n"
        "## Your task\n"
        "Propose exactly ONE change to config.yaml to improve val_score.\n"
        "Follow the Response Format in the Protocol exactly.\n"
    )


def call_qwen(model_name: str, prompt: str) -> str:
    from mlx_lm import generate, load  # noqa: PLC0415

    model, tokenizer = load(model_name)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=formatted, max_tokens=1024, verbose=False)


def parse_response(response: str) -> tuple[str, str | None]:
    """Extract (hypothesis, new_config_yaml) from Qwen's response."""
    hypothesis_match = re.search(r"HYPOTHESIS:\s*(.+)", response)
    hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else "no hypothesis"

    config_match = re.search(
        r"CONFIG_YAML:\s*```ya?ml\s*(.*?)```", response, re.DOTALL
    )
    new_config = config_match.group(1).strip() if config_match else None
    return hypothesis, new_config


def run_training() -> tuple[float | None, float, float]:
    """Run train.py; return (val_score, sensitivity, specificity). None on failure."""
    result = subprocess.run(TRAIN_CMD, capture_output=True, text=True)
    output = result.stdout + result.stderr

    def extract(key: str) -> float:
        m = re.search(rf"{key}:\s*([0-9.]+)", output)
        return float(m.group(1)) if m else 0.0

    val_score_match = re.search(r"val_score:\s*([0-9.]+)", output)
    if not val_score_match:
        print(f"WARNING: could not extract val_score from output:\n{output[:500]}")
        return None, 0.0, 0.0

    return (
        float(val_score_match.group(1)),
        extract("sensitivity"),
        extract("specificity"),
    )


def git_commit(message: str) -> str:
    """Stage config.yaml, commit, return short hash."""
    subprocess.run(["git", "add", "config.yaml"], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def git_revert() -> None:
    subprocess.run(["git", "revert", "HEAD", "--no-edit"], check=True)


def get_best_val_score() -> float:
    if not RESULTS_FILE.exists():
        return 0.0
    lines = RESULTS_FILE.read_text().strip().splitlines()[1:]  # skip header
    scores = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                scores.append(float(parts[1]))
            except ValueError:
                pass
    return max(scores) if scores else 0.0


def append_result(
    commit_hash: str,
    val_score: float,
    sensitivity: float,
    specificity: float,
    description: str,
    status: str,
) -> None:
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "commit_hash\tval_score\tsensitivity\tspecificity\tdescription\tstatus\n"
        )
    with RESULTS_FILE.open("a") as f:
        f.write(
            f"{commit_hash}\t{val_score:.4f}\t{sensitivity:.4f}\t"
            f"{specificity:.4f}\t{description}\t{status}\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoResearch runner — local Qwen on Apple Silicon (mlx-lm)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"MLX model name on Hugging Face (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N experiments (default: 0 = run forever)",
    )
    args = parser.parse_args()

    print(f"AutoResearch runner")
    print(f"  model : {args.model}")
    print(f"  target: {RESULTS_FILE}")
    print(f"  limit : {'unlimited' if not args.max_experiments else args.max_experiments}")
    print("Press Ctrl+C to stop.\n")

    experiment_count = 0
    while True:
        if args.max_experiments and experiment_count >= args.max_experiments:
            print(f"Reached max experiments ({args.max_experiments}). Done.")
            break

        experiment_count += 1
        best = get_best_val_score()
        print(f"\n{'='*60}")
        print(f"Experiment #{experiment_count}  [{datetime.now().strftime('%H:%M:%S')}]")
        print(f"Best val_score so far: {best:.4f}")
        print(f"{'='*60}")

        program_md = read_file("program.md")
        config_yaml = read_file("config.yaml")
        results_history = read_file(str(RESULTS_FILE))

        print("Querying Qwen for proposal...")
        prompt = build_prompt(program_md, config_yaml, results_history)
        try:
            response = call_qwen(args.model, prompt)
        except Exception as exc:
            print(f"ERROR: Qwen call failed: {exc}")
            break

        hypothesis, new_config = parse_response(response)
        print(f"Hypothesis: {hypothesis}")

        if not new_config:
            print("WARNING: could not parse CONFIG_YAML block. Skipping.")
            continue

        try:
            yaml.safe_load(new_config)
        except yaml.YAMLError as exc:
            print(f"WARNING: Qwen produced invalid YAML ({exc}). Skipping.")
            continue

        Path("config.yaml").write_text(new_config)
        commit_hash = git_commit(f"experiment: {hypothesis}")
        print(f"Committed: {commit_hash}")

        print("Running training/train.py...")
        val_score, sensitivity, specificity = run_training()

        if val_score is None:
            print("Training failed — reverting.")
            git_revert()
            continue

        print(f"val_score={val_score:.4f}  sensitivity={sensitivity:.4f}  specificity={specificity:.4f}")

        if val_score > best:
            status = "keep"
            print(f"IMPROVED ({val_score:.4f} > {best:.4f}) — keeping")
        else:
            status = "revert"
            print(f"No improvement ({val_score:.4f} <= {best:.4f}) — reverting")
            git_revert()

        append_result(commit_hash, val_score, sensitivity, specificity, hypothesis, status)

    print("\nAutoResearch loop finished.")


if __name__ == "__main__":
    main()
