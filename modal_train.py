"""Modal training scripts for LeWorldModel.

Usage:
    modal run modal_train.py --wandb-api-key <key>                                        # standard training (cube)
    modal run modal_train.py --no-download --wandb-api-key <key>                        # skip dataset download
    modal run modal_train.py --adversarial --wandb-api-key <key>                        # adversarial fine-tune (cube)
    modal run modal_train.py --adversarial --env-name reacher --wandb-api-key <key>     # adversarial fine-tune (reacher)
    modal run modal_train.py::download_dataset                                  # dataset only
    modal run modal_train.py::download_checkpoint                               # checkpoint only
"""

import modal

app = modal.App("stable-worldmodel-train")

volume = modal.Volume.from_name("stableworldmodel")
STABLEWM_HOME = "/root/.stable_worldmodel"
CUBE_HF_REPO = "quentinll/lewm-cube"
DATASET_NAME = "ogbench/cube_single_expert"
CHECKPOINT_NAME = "lewm-cube"

DATA_CONFIG_DEFAULTS = {
    "ogb": "ogbench/cube_single_expert",
    "dmc": "dmc/reacher_random",
}

ENV_CONFIGS = {
    "cube": dict(
        data_config="ogb",
        run_name="lewm-ogbcube",
        run_name_adv="lewm-ogbcube-adv",
        pretrained="lewm-cube",
    ),
    "reacher": dict(
        data_config="dmc",
        run_name="lewm-reacher",
        run_name_adv="lewm-reacher-adv",
        pretrained="lewm-reacher",
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

_GPU_TRAIN_KWARGS = dict(
    image=image,
    volumes={STABLEWM_HOME: volume},
    gpu="A100-40GB",
    timeout=3600 * 24,
    cpu=20.0,
)


def _make_env(wandb_api_key: str = "", mujoco_gl: bool = False) -> dict:
    import os

    env = {**os.environ, "STABLEWM_HOME": STABLEWM_HOME}
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key
    if mujoco_gl:
        env["MUJOCO_GL"] = "egl"
    return env


def _copy_dataset_local(src, dst: str = "/tmp/dataset.h5") -> str:
    import shutil
    from pathlib import Path

    src, dst = Path(src), Path(dst)
    if not dst.exists():
        print(f"Copying {src} ({src.stat().st_size / 1e9:.1f} GB) to local disk...")
        shutil.copy2(src, dst)
        print("Dataset copy done.")
    return str(dst)


@app.function(**_GPU_TRAIN_KWARGS)
def train(
    wandb_api_key: str = "",
    run_name: str = "",
    max_epochs: int = 100,
    num_workers: int = 16,
    prefetch_factor: int = 8,
    adversarial: bool = False,
    data_config: str = "ogb",
    dataset_name: str = "",
    pretrained_model_name: str = "",
):
    """Train or adversarially fine-tune LeWorldModel."""
    assert wandb_api_key, "wandb_api_key is required — pass --wandb-api-key <key>"
    import subprocess
    from pathlib import Path

    effective_name = dataset_name or DATA_CONFIG_DEFAULTS.get(data_config)
    if not effective_name:
        raise ValueError(f"Cannot resolve dataset for data_config={data_config!r}; pass dataset_name")

    local_h5 = _copy_dataset_local(Path(STABLEWM_HOME) / "datasets" / f"{effective_name}.h5")

    script = "scripts/train/lewm_adversarial.py" if adversarial else "scripts/train/lewm.py"
    cmd = [
        "python", script,
        f"data={data_config}",
        f"data.dataset.path={local_h5}",
        f"output_model_name={run_name}",
        f"trainer.max_epochs={max_epochs}",
        f"num_workers={num_workers}",
        f"loader.prefetch_factor={prefetch_factor}",
        "wandb.enabled=true",
        "wandb.config.entity=null",
        f"hydra.run.dir={STABLEWM_HOME}/hydra_outputs",
    ]
    if adversarial:
        cmd.append(f"pretrained={pretrained_model_name}")

    print("Training:", " ".join(cmd))
    subprocess.run(cmd, cwd="/app", env=_make_env(wandb_api_key), check=True)
    volume.commit()

@app.local_entrypoint()
def main(
    wandb_api_key: str = "",
    env_name: str = "cube",
    run_name: str = "",
    max_epochs: int = 100,
    num_workers: int = 16,
    prefetch_factor: int = 8,
    adversarial: bool = False,
    pretrained_model_name: str = "",
    download: bool = True,
):
    """Train or adversarially fine-tune LeWorldModel.

    Standard training (cube):        modal run modal_train.py --wandb-api-key <key>
    Adversarial fine-tune (cube):    modal run modal_train.py --adversarial --wandb-api-key <key>
    Adversarial fine-tune (reacher): modal run modal_train.py --adversarial --env-name reacher --wandb-api-key <key>
    """
    assert wandb_api_key, "wandb_api_key is required — pass --wandb-api-key <key>"
    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown env_name={env_name!r}; choose from {list(ENV_CONFIGS)}")

    cfg = ENV_CONFIGS[env_name]
    effective_run_name = run_name or (cfg["run_name_adv"] if adversarial else cfg["run_name"])

    train.remote(
        wandb_api_key=wandb_api_key,
        run_name=effective_run_name,
        max_epochs=max_epochs,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        adversarial=adversarial,
        data_config=cfg["data_config"],
        pretrained_model_name=pretrained_model_name or (cfg["pretrained"] if adversarial else ""),
    )
