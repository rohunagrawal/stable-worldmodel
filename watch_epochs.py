"""Watch adversarial training log and run checkpoint comparisons after each epoch.

Usage:
    python watch_epochs.py                                    # reacher (default)
    python watch_epochs.py --env pusht                        # pusht
    python watch_epochs.py --env pusht --start-epoch 3        # skip already-done epochs
    python watch_epochs.py --log-file /tmp/my_train.log       # custom log file
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
REPO_ROOT = Path(__file__).parent

ENV_CONFIGS = {
    "reacher": dict(
        log_file="/tmp/reacher_adv_train.log",
        policy_adv="lewm-reacher-adv",
        policy_base="lewm-reacher",
        env_name="reacher",
        results_md=REPO_ROOT / "eval_results" / "reacher_comparison.md",
        seeds=[42, 123, 456, 789, 1024],
        header=(
            "| Epoch | ADV Mean | ADV Std | ADV Rates | Base Mean | Base Std | Base Rates |\n"
            "|------:|---------:|--------:|-----------|----------:|--------:|------------|\n"
        ),
    ),
    "pusht": dict(
        log_file="/tmp/pusht_adv_train.log",
        policy_adv="lewm-pusht-adv",
        policy_base="lewm-pusht",
        env_name="pusht",
        results_md=REPO_ROOT / "eval_results" / "pusht_comparison.md",
        seeds=[42, 123, 456, 789, 1024],
        header=(
            "| Epoch | ADV Mean | ADV Std | ADV Rates | Base Mean | Base Std | Base Rates |\n"
            "|------:|---------:|--------:|-----------|----------:|--------:|------------|\n"
        ),
    ),
}


def init_md(cfg: dict, results_md: Path):
    results_md.parent.mkdir(parents=True, exist_ok=True)
    if not results_md.exists():
        results_md.write_text(
            f"# {cfg['env_name'].capitalize()} Checkpoint Comparison\n\n"
            f"- **ADV**: `{cfg['policy_adv']}` (adversarially fine-tuned)\n"
            f"- **Base**: `{cfg['policy_base']}` (pretrained)\n"
            f"- **Seeds**: {cfg['seeds']}\n"
            f"- **Solver**: adam (n_steps=30, lr=0.1, horizon=5)\n"
            f"- **num_eval**: 50\n\n"
            + cfg["header"]
        )


def append_row(epoch: int, adv_rates: list, base_rates: list, results_md: Path):
    adv_mean = np.mean(adv_rates) if adv_rates else float("nan")
    adv_std = np.std(adv_rates) if adv_rates else float("nan")
    base_mean = np.mean(base_rates) if base_rates else float("nan")
    base_std = np.std(base_rates) if base_rates else float("nan")

    adv_fmt = [f"{r:.0f}%" for r in adv_rates]
    base_fmt = [f"{r:.0f}%" for r in base_rates]

    with open(results_md, "a") as f:
        f.write(
            f"| {epoch} | {adv_mean:.1f}% | {adv_std:.1f}% | {', '.join(adv_fmt)} "
            f"| {base_mean:.1f}% | {base_std:.1f}% | {', '.join(base_fmt)} |\n"
        )
    print(f"[{datetime.now():%H:%M:%S}] Updated {results_md}")


def run_compare(epoch: int, cfg: dict, results_md: Path):
    print(f"\n{'='*60}")
    print(f"[{datetime.now():%H:%M:%S}] Epoch {epoch} done — launching evals...")

    result = subprocess.run(
        [
            "modal", "run", "modal_plan.py::compare",
            f"--epoch={epoch}",
            f"--policy-adv={cfg['policy_adv']}",
            f"--policy-base={cfg['policy_base']}",
            f"--env-name={cfg['env_name']}",
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
                adv = data["results"].get(cfg["policy_adv"], {})
                base = data["results"].get(cfg["policy_base"], {})
                append_row(epoch, adv.get("rates", []), base.get("rates", []), results_md)
            except Exception as e:
                print(f"Failed to parse RESULTS_JSON: {e}", file=sys.stderr)
            return

    print("WARNING: No RESULTS_JSON line found in output", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="reacher", choices=list(ENV_CONFIGS),
                        help="Environment to watch")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Number of epochs already completed before this watcher started")
    parser.add_argument("--log-file", default="",
                        help="Override log file path (default: per-env config)")
    args = parser.parse_args()

    cfg = ENV_CONFIGS[args.env]
    log_file = args.log_file or cfg["log_file"]
    results_md = cfg["results_md"]

    init_md(cfg, results_md)
    epoch = args.start_epoch

    print(f"Watching {log_file} from current EOF (env={args.env}, start_epoch={epoch})...")
    print(f"Results → {results_md}")

    sanity_check_skipped = False
    with open(log_file) as f:
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
                run_compare(epoch, cfg, results_md)


if __name__ == "__main__":
    main()
