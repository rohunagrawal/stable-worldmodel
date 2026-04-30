"""Modal planning script for LeWorldModel.

Usage:
    modal run modal_plan.py --wandb-api-key <key>                                    # CEM planning (cube)
    modal run modal_plan.py --env-name reacher --wandb-api-key <key>                 # CEM planning (reacher)
    modal run modal_plan.py --solver adam --seed 42 --wandb-api-key <key>            # Adam planning (cube)
    modal run modal_plan.py::download_checkpoint                                     # download cube checkpoint
    modal run modal_plan.py::download_checkpoint --env-name reacher                  # download reacher checkpoint
    for seed in 42 123 456; do
        modal run modal_plan.py --solver adam --seed $seed --wandb-api-key <key>
    done
"""

import modal

app = modal.App("stable-worldmodel-plan")

volume = modal.Volume.from_name("stableworldmodel")
STABLEWM_HOME = "/root/.stable_worldmodel"

ENV_CONFIGS = {
    "cube": dict(
        checkpoint_hf_repo="quentinll/lewm-cube",
        checkpoint_name="lewm-cube",
        config_name="cube",
        goal_offset_steps=50,
        results_txt="ogb_cube_results.txt",
        csv_name="results.csv",
        wandb_project="stable-worldmodel-plan",
    ),
    "reacher": dict(
        checkpoint_hf_repo="quentinll/lewm-reacher",
        checkpoint_name="lewm-reacher",
        config_name="reacher",
        goal_offset_steps=25,
        results_txt="dmc_results.txt",
        csv_name="reacher_results.csv",
        wandb_project="stable-worldmodel-plan-reacher",
    ),
    "pusht": dict(
        checkpoint_hf_repo="quentinll/lewm-pusht",
        checkpoint_name="lewm-pusht",
        config_name="pusht",
        goal_offset_steps=25,
        results_txt="pusht_results.txt",
        csv_name="pusht_results.csv",
        wandb_project="stable-worldmodel-plan-pusht",
    ),
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "zstd",
        "ffmpeg",
        "libegl1",
        "libegl-mesa0",
        "libgles2",
        "libglvnd0",
        "libglvnd-dev",
        "libopengl0",
    )
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["train", "env"])
    .pip_install("wandb", "matplotlib")
    .add_local_dir(
        ".",
        "/app",
        copy=True,
        ignore=[
            ".git",
            "**/__pycache__",
            "**/*.pyc",
            ".ruff_cache",
            "outputs",
            "multirun",
            "*.egg-info",
            ".venv",
        ],
    )
    .run_commands("pip install -e '/app[train,env]' --no-deps")
    .env({"STABLEWM_HOME": STABLEWM_HOME})
)


@app.function(
    image=image,
    volumes={STABLEWM_HOME: volume},
    timeout=3600 * 3,
    cpu=2.0,
)
def download_checkpoint(env_name: str = "cube"):
    """Download checkpoint weights+config into checkpoints/ (idempotent)."""
    import urllib.request

    from stable_worldmodel.data.utils import get_cache_dir

    cfg = ENV_CONFIGS[env_name]
    ckpt_dir = get_cache_dir(sub_folder="checkpoints") / cfg["checkpoint_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://huggingface.co/{cfg['checkpoint_hf_repo']}/resolve/main"
    for filename in ("weights.pt", "config.json"):
        dest = ckpt_dir / filename
        if dest.exists():
            print(f"{filename} already present, skipping.")
            continue
        url = f"{base_url}/{filename}"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  → {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    volume.commit()
    print(f"Checkpoint ready at {ckpt_dir}")


@app.function(
    image=image,
    volumes={STABLEWM_HOME: volume},
    timeout=3600 * 4,
    cpu=4.0,
)
def fix_reacher_dataset():
    """Download + extract the reacher dataset, replacing any truncated h5. Idempotent."""
    import subprocess
    from pathlib import Path

    from stable_worldmodel.data.utils import (
        _download,
        _hf_dataset_find_archive,
        get_cache_dir,
    )

    cfg = ENV_CONFIGS["reacher"]
    HF_REPO = cfg["checkpoint_hf_repo"]
    DATASET_LOCAL_NAME = "dmc/reacher_random"
    HF_BASE_URL = "https://huggingface.co"

    datasets_dir = get_cache_dir(sub_folder="datasets")
    local_dir = datasets_dir / HF_REPO.replace("/", "--")
    local_dir.mkdir(parents=True, exist_ok=True)

    archive_name = "reacher.tar.zst"
    archive_path = local_dir / archive_name
    expected_h5 = datasets_dir / f"{DATASET_LOCAL_NAME}.h5"

    if expected_h5.is_symlink():
        expected_h5.unlink()
        print(f"Removed stale symlink: {expected_h5}")

    for h5 in list(local_dir.glob("*.h5")) + list(local_dir.glob("*.hdf5")):
        print(f"Removing existing h5 ({h5.stat().st_size / 1e9:.2f} GB): {h5}")
        h5.unlink()

    if archive_path.exists():
        print(f"Archive already present ({archive_path.stat().st_size / 1e9:.1f} GB), skipping download.")
    else:
        archive_name = _hf_dataset_find_archive(HF_REPO)
        url = f"{HF_BASE_URL}/datasets/{HF_REPO}/resolve/main/{archive_name}"
        archive_path = local_dir / archive_name
        print(f"Downloading {url} ...")
        _download(url, archive_path)
        volume.commit()
        print(f"Archive downloaded: {archive_path.stat().st_size / 1e9:.1f} GB")

    print(f"Extracting {archive_path} into {local_dir} ...")
    result = subprocess.run(
        ["tar", "--use-compress-program=unzstd", "-xf", str(archive_path), "-C", str(local_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Extraction failed")

    h5_files = list(local_dir.glob("*.h5")) + list(local_dir.glob("*.hdf5"))
    if not h5_files:
        raise RuntimeError("No h5 file found after extraction")
    actual_h5 = h5_files[0]
    print(f"Extracted: {actual_h5} ({actual_h5.stat().st_size / 1e9:.1f} GB)")

    archive_path.unlink()
    expected_h5.parent.mkdir(parents=True, exist_ok=True)
    expected_h5.symlink_to(actual_h5)
    print(f"Linked: {expected_h5} → {actual_h5}")

    volume.commit()
    print("Dataset ready.")


@app.function(
    image=image,
    volumes={STABLEWM_HOME: volume},
    timeout=3600 * 4,
    cpu=4.0,
)
def download_pusht_dataset(force_redownload: bool = False):
    """Download + extract the PushT dataset into datasets/pusht_expert_train.h5. Idempotent."""
    import subprocess
    from pathlib import Path

    from stable_worldmodel.data.utils import (
        _download,
        _hf_dataset_find_archive,
        get_cache_dir,
    )

    HF_REPO = "quentinll/lewm-pusht"
    DATASET_LOCAL_NAME = "pusht_expert_train"
    HF_BASE_URL = "https://huggingface.co"

    datasets_dir = get_cache_dir(sub_folder="datasets")
    local_dir = datasets_dir / HF_REPO.replace("/", "--")
    local_dir.mkdir(parents=True, exist_ok=True)

    expected_h5 = datasets_dir / f"{DATASET_LOCAL_NAME}.h5"
    if not force_redownload and expected_h5.exists() and not expected_h5.is_symlink():
        print(f"Dataset already present at {expected_h5} ({expected_h5.stat().st_size / 1e9:.1f} GB), skipping.")
        return
    if expected_h5.is_symlink():
        expected_h5.unlink()

    archive_name = _hf_dataset_find_archive(HF_REPO)
    url = f"{HF_BASE_URL}/datasets/{HF_REPO}/resolve/main/{archive_name}"
    archive_path = local_dir / archive_name

    if force_redownload and archive_path.exists():
        print(f"Force-removing existing archive: {archive_path}")
        archive_path.unlink()

    if not archive_path.exists():
        print(f"Downloading {url} ...")
        _download(url, archive_path)
        volume.commit()
        print(f"Archive downloaded: {archive_path.stat().st_size / 1e9:.1f} GB")
    else:
        print(f"Archive already present ({archive_path.stat().st_size / 1e9:.1f} GB), skipping download.")

    print(f"Extracting {archive_path} ...")
    if archive_name.endswith(".tar.zst"):
        result = subprocess.run(
            ["tar", "--use-compress-program=unzstd", "-xf", str(archive_path), "-C", str(local_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("Extraction failed")
        h5_files = list(local_dir.glob("*.h5")) + list(local_dir.glob("*.hdf5"))
        if not h5_files:
            raise RuntimeError("No h5 file found after extraction")
        actual_h5 = h5_files[0]
    else:
        # .h5.zst — bare zstd-compressed h5 file
        stem = archive_name[: -len(".zst")]  # e.g. pusht_expert_train.h5
        actual_h5 = local_dir / stem
        result = subprocess.run(
            ["zstd", "-d", str(archive_path), "-o", str(actual_h5), "--force"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("Extraction failed")

    print(f"Extracted: {actual_h5} ({actual_h5.stat().st_size / 1e9:.1f} GB)")

    archive_path.unlink()
    expected_h5.parent.mkdir(parents=True, exist_ok=True)
    expected_h5.symlink_to(actual_h5)
    print(f"Linked: {expected_h5} → {actual_h5}")

    volume.commit()
    print("PushT dataset ready.")


@app.function(
    image=image,
    volumes={STABLEWM_HOME: volume},
    gpu="A100-40GB",
    timeout=3600 * 4,
    cpu=8.0,
)
def plan(
    env_name: str = "cube",
    solver: str = "cem",
    seed: int = 42,
    num_eval: int = 50,
    horizon: int = 5,
    goal_offset_steps: int = 0,
    n_steps: int = 30,
    lr: float = 0.1,
    wandb_api_key: str = "",
    wandb_project: str = "",
    policy: str = "",
    solver_batch_size: int = 0,
):
    """Run one seed of MPC planning and append a row to eval_results/<csv_name>."""
    assert wandb_api_key, "wandb_api_key is required — pass --wandb-api-key <key>"
    import csv
    import os
    import re
    import subprocess
    from pathlib import Path

    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown env_name={env_name!r}; choose from {list(ENV_CONFIGS)}")
    cfg = ENV_CONFIGS[env_name]

    effective_goal_offset = goal_offset_steps or cfg["goal_offset_steps"]
    effective_policy = policy or cfg["checkpoint_name"]
    effective_wandb_project = wandb_project or cfg["wandb_project"]

    env = {**os.environ, "STABLEWM_HOME": STABLEWM_HOME, "MUJOCO_GL": "egl"}
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key
        os.environ["WANDB_API_KEY"] = wandb_api_key

    cmd = [
        "python", "scripts/plan/eval_wm.py",
        "--config-name", cfg["config_name"],
        f"policy={effective_policy}",
        f"solver={solver}",
        f"plan_config.horizon={horizon}",
        f"eval.num_eval={num_eval}",
        f"eval.goal_offset_steps={effective_goal_offset}",
        f"seed={seed}",
        f"hydra.run.dir={STABLEWM_HOME}/hydra_outputs/{solver}_seed{seed}",
    ]
    if solver == "adam":
        cmd += [
            f"solver.n_steps={n_steps}",
            f"solver.optimizer_kwargs.lr={lr}",
            "++compile=True",
        ]
    if solver_batch_size > 0:
        cmd += [f"solver.batch_size={solver_batch_size}"]

    print(f"[{env_name}/{solver}/seed={seed}] Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd="/app", env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Eval failed for {env_name}/{solver}/seed={seed}")

    results_txt = Path(STABLEWM_HOME) / "checkpoints" / cfg["results_txt"]
    success_rate = None
    eval_time = None
    peak_gpu_memory_gb = None
    if results_txt.exists():
        txt = results_txt.read_text()
        for m in re.finditer(r"'success_rate':\s*([\d.]+)", txt):
            success_rate = float(m.group(1))
        for m in re.finditer(r"evaluation_time:\s*([\d.]+)", txt):
            eval_time = float(m.group(1))

    for m in re.finditer(r"peak_gpu_memory_gb:\s*([\d.]+)", result.stdout):
        peak_gpu_memory_gb = float(m.group(1))

    print(f"[{env_name}/{solver}/seed={seed}] success_rate={success_rate}  eval_time={eval_time}s  peak_gpu_memory={peak_gpu_memory_gb}GB")

    row = {
        "env": env_name,
        "solver": solver,
        "seed": seed,
        "num_eval": num_eval,
        "horizon": horizon,
        "goal_offset_steps": effective_goal_offset,
        "n_steps": n_steps if solver == "adam" else "",
        "lr": lr if solver == "adam" else "",
        "checkpoint_path": str(Path(STABLEWM_HOME) / "checkpoints" / effective_policy),
        "success_rate": success_rate,
        "eval_time_s": eval_time,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
    }

    output_dir = Path(STABLEWM_HOME) / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / cfg["csv_name"]

    volume.reload()
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    if wandb_api_key:
        import wandb
        wandb.init(
            project=effective_wandb_project,
            name=f"{env_name}-{solver}-seed{seed}",
            config={k: v for k, v in row.items() if k not in ("success_rate", "eval_time_s")},
        )
        wandb.log({"success_rate": success_rate, "eval_time_s": eval_time, "peak_gpu_memory_gb": peak_gpu_memory_gb})
        wandb.finish()

    volume.commit()
    print(f"Row appended to {csv_path}")

    plot_bytes = None
    plot_path = Path(STABLEWM_HOME) / "checkpoints" / f"planning_metrics_{effective_policy}.png"
    if plot_path.exists():
        plot_bytes = plot_path.read_bytes()
        print(f"Planning metrics plot read ({len(plot_bytes)} bytes)")

    return {"success_rate": success_rate, "plot_bytes": plot_bytes}


@app.function(
    image=image,
    volumes={STABLEWM_HOME: volume},
    timeout=120,
    cpu=1.0,
)
def refresh_adv_weights(epoch: int, policy: str = "lewm-reacher-adv"):
    """Copy weights_epoch_N.pt → weights.pt so load_pretrained picks the latest epoch."""
    import shutil
    from stable_worldmodel.data.utils import get_cache_dir

    ckpt_dir = get_cache_dir(sub_folder="checkpoints") / policy
    src = ckpt_dir / f"weights_epoch_{epoch}.pt"
    dst = ckpt_dir / "weights.pt"
    if src.exists():
        shutil.copy2(src, dst)
        volume.commit()
        print(f"Refreshed: {src.name} → weights.pt in {ckpt_dir}")
    else:
        available = sorted(p.name for p in ckpt_dir.glob("weights*.pt"))
        print(f"WARNING: {src.name} not found. Available: {available}")


@app.local_entrypoint()
def compare(
    epoch: int = 0,
    policy_adv: str = "lewm-reacher-adv",
    policy_base: str = "lewm-reacher",
    env_name: str = "reacher",
    wandb_api_key: str = "",
    num_eval: int = 50,
):
    """Run 5 seeds × 2 checkpoints in parallel and print RESULTS_JSON."""
    import json
    import numpy as np

    seeds = [42, 123, 456, 789, 1024]

    # Refresh adv checkpoint so load_pretrained picks epoch N weights
    refresh_adv_weights.remote(epoch=epoch, policy=policy_adv)

    # Spawn all 10 evals in parallel
    handles = []
    for policy in [policy_adv, policy_base]:
        for seed in seeds:
            h = plan.spawn(
                env_name=env_name,
                solver="adam",
                seed=seed,
                num_eval=num_eval,
                wandb_api_key=wandb_api_key,
                policy=policy,
                n_steps=30,
                lr=0.1,
            )
            handles.append((policy, seed, h))

    results = {policy_adv: [], policy_base: []}
    for policy, seed, h in handles:
        try:
            ret = h.get()
            sr = ret["success_rate"] if isinstance(ret, dict) else ret
            if sr is not None:
                results[policy].append(sr)
        except Exception as e:
            print(f"WARNING: eval failed policy={policy} seed={seed}: {e}")

    summary = {}
    for policy, rates in results.items():
        summary[policy] = {
            "mean": round(float(np.mean(rates)), 1) if rates else None,
            "std": round(float(np.std(rates)), 1) if rates else None,
            "rates": rates,
        }

    print(f"RESULTS_JSON:{json.dumps({'epoch': epoch, 'results': summary})}")


@app.local_entrypoint()
def main(
    env_name: str = "cube",
    solver: str = "cem",
    seed: int = 42,
    num_eval: int = 50,
    horizon: int = 5,
    goal_offset_steps: int = 0,
    n_steps: int = 30,
    lr: float = 0.1,
    wandb_api_key: str = "",
    wandb_project: str = "",
    policy: str = "",
    solver_batch_size: int = 0,
):
    """Run one seed of MPC planning and append results to the shared CSV."""
    assert wandb_api_key, "wandb_api_key is required — pass --wandb-api-key <key>"
    from pathlib import Path

    ret = plan.remote(
        env_name=env_name,
        solver=solver,
        seed=seed,
        num_eval=num_eval,
        horizon=horizon,
        goal_offset_steps=goal_offset_steps,
        n_steps=n_steps,
        lr=lr,
        wandb_api_key=wandb_api_key,
        wandb_project=wandb_project,
        policy=policy,
        solver_batch_size=solver_batch_size,
    )
    if isinstance(ret, dict) and ret.get("plot_bytes"):
        _policy_name = policy or ENV_CONFIGS[env_name]["checkpoint_name"]
        local_plot = Path(f"eval_results/planning_metrics_{env_name}_{solver}_{_policy_name}_seed{seed}.png")
        local_plot.parent.mkdir(parents=True, exist_ok=True)
        local_plot.write_bytes(ret["plot_bytes"])
        print(f"Planning metrics plot saved locally to {local_plot}")


# --- Utility functions ---

@app.function(
    image=image,
    volumes={STABLEWM_HOME: volume},
    gpu="A100-40GB",
    timeout=120,
    cpu=4.0,
)
def debug_eval(env_name: str = "cube", policy: str = ""):
    """Run a minimal 2-episode eval with streaming output to diagnose failures."""
    import os
    import subprocess

    from stable_worldmodel.data.utils import get_cache_dir

    cfg = ENV_CONFIGS[env_name]
    effective_policy = policy or cfg["checkpoint_name"]
    ckpt_dir = get_cache_dir(sub_folder="checkpoints") / effective_policy
    print(f"Checkpoint dir contents: {sorted(p.name for p in ckpt_dir.glob('*'))}")

    env = {**os.environ, "STABLEWM_HOME": STABLEWM_HOME, "MUJOCO_GL": "egl"}
    subprocess.run(
        [
            "python", "scripts/plan/eval_wm.py",
            "--config-name", cfg["config_name"],
            f"policy={effective_policy}",
            "solver=adam", "plan_config.horizon=5",
            "eval.num_eval=2", f"eval.goal_offset_steps={cfg['goal_offset_steps'] // 2}",
            "seed=42", "solver.n_steps=3", "solver.optimizer_kwargs.lr=0.1",
            f"hydra.run.dir={STABLEWM_HOME}/hydra_outputs/debug",
        ],
        cwd="/app", env=env,
    )


@app.function(
    image=image,
    volumes={STABLEWM_HOME: volume},
    timeout=60,
    cpu=1.0,
)
def copy_checkpoint_config(src: str = "lewm-cube", dst: str = "lewm-ogbcube-adv"):
    """Copy config.json from src checkpoint to dst so load_pretrained() can instantiate it."""
    import shutil

    from stable_worldmodel.data.utils import get_cache_dir

    ckpt_dir = get_cache_dir(sub_folder="checkpoints")
    src_path = ckpt_dir / src / "config.json"
    dst_path = ckpt_dir / dst / "config.json"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, dst_path)
    print(f"Copied {src_path} → {dst_path}")
    volume.commit()


@app.function(
    image=image,
    volumes={STABLEWM_HOME: volume},
    timeout=300,
    cpu=2.0,
)
def dataset_stats(dataset_name: str = "ogbench/cube_single_expert"):
    """Print episode length stats for a dataset."""
    from pathlib import Path

    import numpy as np

    import stable_worldmodel as swm

    dataset = swm.data.HDF5Dataset(dataset_name, keys_to_cache=[], cache_dir=Path(STABLEWM_HOME))
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices = np.unique(dataset.get_col_data(col_name))
    step_idx = dataset.get_col_data("step_idx")
    ep_idx_data = dataset.get_col_data(col_name)

    lengths = np.array([int(np.max(step_idx[ep_idx_data == ep])) + 1 for ep in ep_indices])
    print(f"Num episodes : {len(lengths)}")
    print(f"Min length   : {lengths.min()}")
    print(f"Max length   : {lengths.max()}")
    print(f"Mean length  : {lengths.mean():.1f}")
    print(f"Median length: {int(np.median(lengths))}")
    print(f"Episodes >= 50 steps : {(lengths >= 50).sum()} / {len(lengths)}")
    print(f"Episodes >= 25 steps : {(lengths >= 25).sum()} / {len(lengths)}")
