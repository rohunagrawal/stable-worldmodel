"""Modal script to generate PushT loss landscape data for gradient.py visualization.

Produces the same file format as testing/pusht.* for both the pretrained and
adversarial PushT world models:
  testing/pusht2.pretrained/{A,B}.npy  Z.pth  markers.npy  alphas_path.npy  betas_path.npy
  testing/pusht2.adv/{A,B}.npy         Z.pth  markers.npy  alphas_path.npy  betas_path.npy

Axes are shared between both models:
  α  direction: a_GT → a_base_plan   (pretrained WM's optimal action)
  β  direction: a_GT → a_adv_plan    (adversarial WM's optimal action)

Usage:
    modal run modal_landscape_pusht.py
    modal run modal_landscape_pusht.py --epoch-adv 3 --grid-size 50
"""

import modal

app = modal.App("stable-worldmodel-landscape-pusht")

volume = modal.Volume.from_name("stableworldmodel")
STABLEWM_HOME = "/root/.stable_worldmodel"

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
    gpu="A100-40GB",
    timeout=3600 * 2,
    cpu=4.0,
)
def generate_landscape(
    epoch_adv: int = 3,
    grid_size: int = 50,
    n_gd_steps: int = 30,
    episode_seed: int = 42,
    batch_size_grid: int = 100,
):
    """Generate the loss landscape for both PushT models and return data + plot bytes."""
    import os
    import shutil
    import numpy as np
    import torch
    from pathlib import Path
    from torchvision.transforms import v2 as transforms
    from torchvision import tv_tensors
    from sklearn import preprocessing
    from gymnasium.spaces import Box
    import dataclasses

    os.environ["MUJOCO_GL"] = "egl"

    import stable_worldmodel as swm
    import stable_pretraining as spt
    from stable_worldmodel.solver.gd import GradientSolver

    device = "cuda"

    # ── Constants ────────────────────────────────────────────────────────────────
    HORIZON = 5
    ACTION_BLOCK = 5
    SINGLE_ACTION_DIM = 2
    ACTION_DIM_FLAT = ACTION_BLOCK * SINGLE_ACTION_DIM  # = 10
    GOAL_OFFSET = 25

    # ── Dataset ──────────────────────────────────────────────────────────────────
    dataset_path = Path(STABLEWM_HOME)
    dataset = swm.data.HDF5Dataset(
        "pusht_expert_train",
        keys_to_cache=["action"],
        cache_dir=dataset_path,
    )

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_idx_data = dataset.get_col_data(col_name)
    step_idx_data = dataset.get_col_data("step_idx")
    ep_indices = np.unique(ep_idx_data)

    action_scaler = preprocessing.StandardScaler()
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    action_scaler.fit(action_data)

    ep_lens = np.array(
        [int(np.max(step_idx_data[ep_idx_data == ep])) + 1 for ep in ep_indices]
    )
    valid_eps = ep_indices[ep_lens >= GOAL_OFFSET + HORIZON * ACTION_BLOCK + 1]
    rng = np.random.default_rng(episode_seed)
    ep_id = int(rng.choice(valid_eps))
    print(f"Using episode {ep_id} (seed={episode_seed})")

    ep_data = dataset.load_chunk(
        np.array([ep_id]),
        np.array([0]),
        np.array([GOAL_OFFSET + 1]),
    )[0]

    # ── Image transform ───────────────────────────────────────────────────────────
    img_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])

    def process_frame(chw_uint8_tensor):
        return img_transform(tv_tensors.Image(chw_uint8_tensor))

    obs_frame = ep_data["pixels"][0]
    goal_frame = ep_data["pixels"][GOAL_OFFSET]

    obs_pix = process_frame(obs_frame).unsqueeze(0).unsqueeze(0).to(device)
    goal_pix = process_frame(goal_frame).unsqueeze(0).unsqueeze(0).to(device)

    # ── GT action sequence ───────────────────────────────────────────────────────
    a_gt_raw = ep_data["action"][:GOAL_OFFSET].numpy()       # (25, 2)
    a_gt_norm = action_scaler.transform(a_gt_raw)            # (25, 2)
    a_gt = torch.from_numpy(
        a_gt_norm.reshape(HORIZON, ACTION_DIM_FLAT)
    ).float()                                                  # (5, 10)

    # ── Load models ───────────────────────────────────────────────────────────────
    stablewm_home = Path(STABLEWM_HOME)
    ckpt_dir = stablewm_home / "checkpoints"

    model_base = swm.wm.utils.load_pretrained("lewm-pusht", cache_dir=stablewm_home)
    model_base = model_base.to(device).eval()
    model_base.requires_grad_(False)

    adv_src = ckpt_dir / "lewm-pusht-adv" / f"weights_epoch_{epoch_adv}.pt"
    adv_dst = ckpt_dir / "lewm-pusht-adv" / "weights.pt"
    if adv_src.exists():
        shutil.copy2(adv_src, adv_dst)
        print(f"Copied {adv_src.name} → weights.pt")
    else:
        print(f"No weights_epoch_{epoch_adv}.pt found; using existing weights.pt")
    model_adv = swm.wm.utils.load_pretrained("lewm-pusht-adv", cache_dir=stablewm_home)
    model_adv = model_adv.to(device).eval()
    model_adv.requires_grad_(False)

    # ── GD solver helper ─────────────────────────────────────────────────────────
    @dataclasses.dataclass
    class _Config:
        horizon: int
        action_block: int
        receding_horizon: int = HORIZON
        history_len: int = 1
        warm_start: bool = True

    mock_action_space = Box(-1.0, 1.0, shape=(1, SINGLE_ACTION_DIM), dtype=np.float32)
    config = _Config(horizon=HORIZON, action_block=ACTION_BLOCK)

    def run_gd(model, obs_pix_dev, goal_pix_dev):
        solver = GradientSolver(
            model=model,
            n_steps=n_gd_steps,
            batch_size=1,
            num_samples=1,
            var_scale=1.0,
            device=device,
            seed=42,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.1},
        )
        solver.configure(action_space=mock_action_space, n_envs=1, config=config)

        info = {
            "pixels": obs_pix_dev,
            "goal": goal_pix_dev,
            "action": torch.zeros(1, 1, ACTION_DIM_FLAT, device=device),
        }

        with torch.enable_grad():
            outputs = solver.solve(info)

        a_plan = outputs["actions"].cpu()[0]
        all_step_actions = outputs["step_metrics"]["all_step_actions"]
        return a_plan, all_step_actions

    print("Running GD for base model…")
    a_base, steps_base = run_gd(model_base, obs_pix, goal_pix)
    print("Running GD for adv model…")
    a_adv, steps_adv = run_gd(model_adv, obs_pix.clone(), goal_pix.clone())

    # ── Shared 2-D basis ─────────────────────────────────────────────────────────
    a_gt_flat = a_gt.numpy().flatten()
    d_alpha = (a_base - a_gt).numpy().flatten()
    d_beta = (a_adv - a_gt).numpy().flatten()
    basis = np.stack([d_alpha, d_beta], axis=1)

    # ── Build grid ────────────────────────────────────────────────────────────────
    alphas = np.linspace(-1.5, 1.5, grid_size)
    betas = np.linspace(-1.5, 1.5, grid_size)
    A_grid, B_grid = np.meshgrid(alphas, betas)

    N = grid_size * grid_size
    alpha_flat = A_grid.flatten()
    beta_flat = B_grid.flatten()
    a_grid_flat = (
        a_gt_flat[None, :]
        + alpha_flat[:, None] * d_alpha[None, :]
        + beta_flat[:, None] * d_beta[None, :]
    )
    a_grid = (
        torch.from_numpy(a_grid_flat.reshape(N, HORIZON, ACTION_DIM_FLAT))
        .float()
        .to(device)
    )

    # ── Grid cost evaluation ─────────────────────────────────────────────────────
    def eval_grid_costs(model, obs_pix_dev, goal_pix_dev):
        with torch.no_grad():
            goal_emb = model.encode({"pixels": goal_pix_dev})["emb"]
            obs_emb = model.encode({"pixels": obs_pix_dev})["emb"]

        costs_all = []
        for start in range(0, N, batch_size_grid):
            bs_actions = a_grid[start : start + batch_size_grid]
            bs = bs_actions.shape[0]
            action_cands = bs_actions.unsqueeze(1)
            obs_pix_expanded = obs_pix_dev.unsqueeze(1).expand(bs, 1, -1, -1, -1, -1)
            info = {
                "pixels": obs_pix_expanded,
                "emb": obs_emb.unsqueeze(1).expand(bs, 1, -1, -1).contiguous(),
                "goal_emb": goal_emb.expand(bs, -1, -1).contiguous(),
            }
            with torch.no_grad():
                info = model.rollout(info, action_cands)
                cost = model.criterion(info)
            costs_all.append(cost[:, 0].cpu())

        return torch.cat(costs_all).reshape(grid_size, grid_size).numpy()

    print("Evaluating base model on grid…")
    Z_base = eval_grid_costs(model_base, obs_pix, goal_pix)
    print("Evaluating adv model on grid…")
    Z_adv = eval_grid_costs(model_adv, obs_pix.clone(), goal_pix.clone())

    print(f"Z_base range: [{Z_base.min():.4f}, {Z_base.max():.4f}]")
    print(f"Z_adv  range: [{Z_adv.min():.4f},  {Z_adv.max():.4f}]")

    # ── Project GD paths ─────────────────────────────────────────────────────────
    def project_path(all_step_actions):
        alphas_p, betas_p = [], []
        for step_acts in all_step_actions:
            a_step = step_acts[0, 0].numpy().flatten()
            diff = a_step - a_gt_flat
            coeffs, _, _, _ = np.linalg.lstsq(basis, diff, rcond=None)
            alphas_p.append(float(coeffs[0]))
            betas_p.append(float(coeffs[1]))
        return np.array(alphas_p), np.array(betas_p)

    alphas_path_base, betas_path_base = project_path(steps_base)
    alphas_path_adv, betas_path_adv = project_path(steps_adv)

    diff_init = -a_gt_flat
    coeffs_init, _, _, _ = np.linalg.lstsq(basis, diff_init, rcond=None)
    alpha_init, beta_init = float(coeffs_init[0]), float(coeffs_init[1])

    print(f"alpha_init={alpha_init:.3f}  beta_init={beta_init:.3f}")
    print(f"alphas_path_base: {alphas_path_base[:5]}…")
    print(f"alphas_path_adv:  {alphas_path_adv[:5]}…")

    # ── Save data files ───────────────────────────────────────────────────────────
    models_data = {
        "pusht2.pretrained": dict(
            Z=Z_base,
            markers=np.array([alpha_init, beta_init, 1.0, 0.0, 0.0, 0.0]),
            alphas_path=alphas_path_base,
            betas_path=betas_path_base,
        ),
        "pusht2.adv": dict(
            Z=Z_adv,
            markers=np.array([alpha_init, beta_init, 0.0, 1.0, 0.0, 0.0]),
            alphas_path=alphas_path_adv,
            betas_path=betas_path_adv,
        ),
    }

    testing_dir = Path("/app/testing")
    testing_dir.mkdir(exist_ok=True)

    for model_name, data in models_data.items():
        d = testing_dir / model_name
        d.mkdir(exist_ok=True)
        np.save(d / "A.npy", A_grid)
        np.save(d / "B.npy", B_grid)
        torch.save(torch.from_numpy(data["Z"]).float(), d / "Z.pth")
        np.save(d / "markers.npy", data["markers"])
        np.save(d / "alphas_path.npy", data["alphas_path"])
        np.save(d / "betas_path.npy", data["betas_path"])
        print(f"Saved {d}")

    # ── Generate landscape plot ───────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import PowerNorm

    TICKS = [-1, -0.5, 0, 0.5, 1, 1.5]
    Z_min_global = min(Z_base.min(), Z_adv.min())
    Z_vis_max = float(np.percentile(np.concatenate([Z_base.flatten(), Z_adv.flatten()]), 95))
    norm = PowerNorm(gamma=0.6, vmin=Z_min_global, vmax=Z_vis_max)
    cmap = cm.viridis

    plot_specs = [
        ("Pretrained WM", Z_base, alphas_path_base, betas_path_base,
         models_data["pusht2.pretrained"]["markers"]),
        ("Adversarial WM", Z_adv, alphas_path_adv, betas_path_adv,
         models_data["pusht2.adv"]["markers"]),
    ]

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0], projection="3d"),
        fig.add_subplot(gs[1, 1], projection="3d"),
    ]

    for ax, (name, Z, ap, bp, markers) in zip(axes[:2], plot_specs):
        levels = np.linspace(Z_min_global, Z_vis_max, 40)
        cs = ax.contour(A_grid, B_grid, Z, levels=levels, linewidths=0.5, cmap=cmap, norm=norm)
        ax.clabel(cs, inline=True, fontsize=6)
        ax.plot(ap, bp, "r-", linewidth=1.5, alpha=0.8, label="GD path")
        ax.scatter(*markers[:2], marker="o", color="#e41a1c", zorder=5, s=60)
        ax.scatter(*markers[2:4], marker="x", color="#e41a1c", zorder=5, s=100)
        ax.scatter(*markers[4:6], marker="*", color="#ffbf00", zorder=5, s=120, label="GT")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks(TICKS)
        ax.set_yticks(TICKS)
        ax.set_title(name, fontsize=16, pad=10)

    for ax, (name, Z, ap, bp, markers) in zip(axes[2:], plot_specs):
        Z_clipped = np.clip(Z, Z_min_global, Z_vis_max)
        ax.plot_surface(
            A_grid, B_grid, Z_clipped, cmap=cmap, norm=norm,
            rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=1.0,
        )
        ax.set_box_aspect([1, 1, 0.8])
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks(TICKS)
        ax.set_yticks(TICKS)
        ax.set_zlim(Z_min_global, Z_vis_max)
        ax.set_xlabel(r"$\alpha$", labelpad=4)
        ax.set_ylabel(r"$\beta$", labelpad=4)
        ax.view_init(elev=30, azim=-45)

    cbar_ax = fig.add_axes([0.945, 0.49 - 0.215, 0.018, 0.43])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cbar_ax, label=r"Loss ($\gamma$-scaled; $\gamma$=0.6, capped p95)")

    fig.text(
        0.5, 0.035,
        r"$\alpha : a_{GT} \to \hat{a}_\mathrm{Pretrained}$",
        ha="center", va="center", fontsize=16,
    )
    fig.text(
        0.035, 0.5,
        r"$\beta : a_{GT} \to \hat{a}_\mathrm{Adversarial}$",
        ha="center", va="center", rotation="vertical", fontsize=16,
    )
    plt.subplots_adjust(
        left=0.08, right=0.93, top=0.92, bottom=0.07, wspace=0.00, hspace=0.1
    )

    plot_path = testing_dir / "pusht_landscape.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

    # ── Bundle results as tar archive for download ────────────────────────────────
    import io, tarfile

    results = {}
    if plot_path.exists():
        results["plot_bytes"] = plot_path.read_bytes()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for model_name in models_data:
            tar.add(testing_dir / model_name, arcname=model_name)
    results["tar_bytes"] = buf.getvalue()

    return results


@app.local_entrypoint()
def main(
    epoch_adv: int = 3,
    grid_size: int = 50,
    n_gd_steps: int = 30,
    episode_seed: int = 42,
):
    import io, tarfile
    from pathlib import Path

    print(
        f"Generating PushT loss landscape "
        f"(grid={grid_size}×{grid_size}, gd_steps={n_gd_steps}, epoch_adv={epoch_adv})…"
    )
    ret = generate_landscape.remote(
        epoch_adv=epoch_adv,
        grid_size=grid_size,
        n_gd_steps=n_gd_steps,
        episode_seed=episode_seed,
    )

    if ret.get("plot_bytes"):
        local_plot = Path("eval_results/pusht_landscape.png")
        local_plot.parent.mkdir(parents=True, exist_ok=True)
        local_plot.write_bytes(ret["plot_bytes"])
        print(f"Plot → {local_plot}")

    if ret.get("tar_bytes"):
        out_base = Path("testing")
        out_base.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(ret["tar_bytes"]), mode="r:gz") as tar:
            tar.extractall(path=out_base)
        for model_name in ("pusht2.pretrained", "pusht2.adv"):
            print(f"Data → {out_base / model_name}/")
