"""Watch reacher adversarial training log and run checkpoint comparisons after each epoch.

Usage:
    python watch_epochs.py
    python watch_epochs.py --start-epoch 3  # if N epochs already completed
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

WANDB_KEY = "817eef85fffa99514f7e929bfd5068f2eb349c56"
LOG_FILE = "/tmp/reacher_adv_train.log"
RESULTS_MD = Path(__file__).parent / "eval_results" / "reacher_comparison.md"
REPO_ROOT = Path(__file__).parent

POLICY_ADV = "lewm-reacher-adv"
POLICY_BASE = "lewm-reacher"
SEEDS = [42, 123, 456, 789, 1024]


def init_md():
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_MD.exists():
        RESULTS_MD.write_text(
            f"# Reacher Checkpoint Comparison\n\n"
            f"- **ADV**: `{POLICY_ADV}` (adversarially fine-tuned)\n"
            f"- **Base**: `{POLICY_BASE}` (pretrained)\n"
            f"- **Seeds**: {SEEDS}\n"
            f"- **Solver**: adam (n_steps=30, lr=0.1, horizon=5)\n"
            f"- **num_eval**: 50\n\n"
            f"| Epoch | ADV Mean | ADV Std | ADV Rates | Base Mean | Base Std | Base Rates |\n"
            f"|------:|---------:|--------:|-----------|----------:|--------:|------------|\n"
        )


def append_row(epoch: int, adv_rates: list, base_rates: list):
    adv_mean = np.mean(adv_rates) if adv_rates else float("nan")
    adv_std = np.std(adv_rates) if adv_rates else float("nan")
    base_mean = np.mean(base_rates) if base_rates else float("nan")
    base_std = np.std(base_rates) if base_rates else float("nan")

    adv_fmt = [f"{r:.0f}%" for r in adv_rates]
    base_fmt = [f"{r:.0f}%" for r in base_rates]

    with open(RESULTS_MD, "a") as f:
        f.write(
            f"| {epoch} | {adv_mean:.1f}% | {adv_std:.1f}% | {', '.join(adv_fmt)} "
            f"| {base_mean:.1f}% | {base_std:.1f}% | {', '.join(base_fmt)} |\n"
        )
    print(f"[{datetime.now():%H:%M:%S}] Updated {RESULTS_MD}")


def run_compare(epoch: int):
    print(f"\n{'='*60}")
    print(f"[{datetime.now():%H:%M:%S}] Epoch {epoch} done — launching 10 evals...")

    result = subprocess.run(
        [
            "modal", "run", "modal_plan.py::compare",
            f"--epoch={epoch}",
            f"--policy-adv={POLICY_ADV}",
            f"--policy-base={POLICY_BASE}",
            f"--wandb-api-key={WANDB_KEY}",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    if result.stdout:
        print(result.stdout[-4000:])
    if result.returncode != 0:
        print(f"COMPARE FAILED (exit {result.returncode}):\n{result.stderr[-2000:]}", file=sys.stderr)
        return

    for line in result.stdout.splitlines():
        if line.startswith("RESULTS_JSON:"):
            try:
                data = json.loads(line[len("RESULTS_JSON:"):])
                adv = data["results"].get(POLICY_ADV, {})
                base = data["results"].get(POLICY_BASE, {})
                append_row(epoch, adv.get("rates", []), base.get("rates", []))
            except Exception as e:
                print(f"Failed to parse RESULTS_JSON: {e}", file=sys.stderr)
            return

    print("WARNING: No RESULTS_JSON line found in output", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Number of epochs already completed before this watcher started")
    args = parser.parse_args()

    init_md()
    epoch = args.start_epoch

    print(f"Watching {LOG_FILE} from current EOF (start_epoch={epoch})...")
    print(f"Results → {RESULTS_MD}")

    sanity_check_skipped = False
    with open(LOG_FILE) as f:
        f.seek(0, 2)  # start from end of current file
        while True:
            line = f.readline()
            if not line:
                time.sleep(2)
                continue
            if "pred_loss_epoch" in line:
                if not sanity_check_skipped:
                    sanity_check_skipped = True
                    print(f"[{datetime.now():%H:%M:%S}] Sanity check detected, skipping.")
                    continue
                epoch += 1
                run_compare(epoch)


if __name__ == "__main__":
    main()
