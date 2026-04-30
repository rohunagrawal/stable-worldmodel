import torch
import numpy as np

def load(model: str, Z_diff=None):
    A = np.load(f"/Users/nimit/Downloads/testing/{model}/A.npy")
    B = np.load(f"/Users/nimit/Downloads/testing/{model}/B.npy")
    Z = np.array(torch.load(f"/Users/nimit/Downloads/testing/{model}/Z.pth"), dtype=np.float64)
    if Z_diff is not None:
        Z -= Z_diff

    α_init, β_init, α_plan, β_plan, α_gt, β_gt = np.load(f"/Users/nimit/Downloads/testing/{model}/markers.npy")
    alphas_path = np.load(f"/Users/nimit/Downloads/testing/{model}/alphas_path.npy")
    betas_path = np.load(f"/Users/nimit/Downloads/testing/{model}/betas_path.npy")

    min_idx = np.nonzero(Z == Z.min())
    i_min, j_min = min_idx[0].item(), min_idx[1].item()
    α_min, β_min = A[i_min, j_min].item(), B[i_min, j_min].item()
    Z_min = Z[i_min, j_min].item()

    return A, B, Z, α_init, β_init, α_plan, β_plan, α_gt, β_gt, α_min, β_min, Z_min, alphas_path, betas_path

MARKERS = {
    "init": dict(marker="o",  color="#e41a1c", label="Initialization"),
    "plan": dict(marker="x",  color="#e41a1c"),
    "gt":   dict(marker="*",  color="#ffbf00", label="Ground Truth"),
}

models = [
    ("S", "Simulator + DINOv2", "pusht.simulator"),
    ("T", "DINO-WM", "pusht.pretrained"),
    ("O", "Online WM", "pusht.dagger.6000"),
    ("A", "Adversarial WM", "pusht.pgd.full"),
]
datas = [load(m) for _, _, m in models]

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{mathptmx} \usepackage[T1]{fontenc}',
    'font.family': 'serif',
})

# === Shared setup ===
last_alpha_betas = [(M, (d[-2][-1], d[-1][-1])) for (M, _, _), d in zip(models, datas)]

Z_all = [d[2] for d in datas]
Z_min_global = min(Z.min() for Z in Z_all)
Z_max_global = max(Z.max() for Z in Z_all)

norm = PowerNorm(gamma=0.6, vmin=Z_min_global, vmax=Z_max_global)
cmap = cm.viridis

# === Plot functions ===
def _crop(data, lim):
    """Crop A, B, Z grids to the given symmetric range."""
    A, B, Z = data[0], data[1], data[2]
    mask_i = (A[:, 0] >= lim[0]) & (A[:, 0] <= lim[1])
    mask_j = (B[0, :] >= lim[0]) & (B[0, :] <= lim[1])
    A = A[np.ix_(mask_i, mask_j)]
    B = B[np.ix_(mask_i, mask_j)]
    Z = Z[np.ix_(mask_i, mask_j)]
    return (A, B, Z) + data[3:]

TICKS = [-1, -0.5, 0, 0.5, 1]

def contour(ax, model_name, model_path, data, last_alpha_beta, norm, cmap, lim=None):
    if lim is not None:
        data = _crop(data, lim)
    A, B, Z, *_ = data
    levels = np.arange(0, 0.5, 0.01)
    cs = ax.contour(A, B, Z, levels=levels, linewidths=0.5, cmap=cmap, norm=norm)
    ax.clabel(cs, inline=True, fontsize=6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.set_title(model_name, fontsize=16, pad=10)
    return cs

def surface(ax, model_name, model_path, data, last_alpha_beta, norm, cmap, fade_beyond=None):
    A, B, Z, *_ = data
    if fade_beyond is not None:
        # Compute facecolors with alpha fade beyond the threshold
        Z_norm = norm(Z)
        rgba = cmap(Z_norm)
        # Face centers are averages of surrounding vertices
        A_face = (A[:-1, :-1] + A[1:, 1:]) / 2
        B_face = (B[:-1, :-1] + B[1:, 1:]) / 2
        dist = np.maximum(np.abs(A_face), np.abs(B_face))
        alpha = np.where(dist <= fade_beyond, 1.0, 0.1)
        face_rgba = cmap(norm((Z[:-1, :-1] + Z[1:, 1:]) / 2))
        face_rgba[..., 3] = alpha
        ax.plot_surface(A, B, Z, facecolors=face_rgba, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False)
    else:
        ax.plot_surface(A, B, Z, cmap=cmap, norm=norm, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, alpha=1.0)
    ax.set_box_aspect([1, 1, 0.8])
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.set_zlim(Z.min(), 0.30)
    ax.set_xlabel(r'$\alpha$', labelpad=4)
    ax.set_ylabel(r'$\beta$', labelpad=4)
    ax.view_init(elev=30, azim=-45)
    return ax

# === Figure + layout (2 rows x 4 cols) ===
ncols = len(models)
fig = plt.figure(figsize=(6 * ncols, 10))
gs = fig.add_gridspec(2, ncols, height_ratios=[1, 1.1])

top_axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
bot_axes = [fig.add_subplot(gs[1, i], projection='3d') for i in range(ncols)]

# === Top row (2D contour plots) ===
for i, (ax, (letter, model_name, model_path), data) in enumerate(zip(top_axes, models, datas)):
    contour(ax, model_name, model_path, data, last_alpha_betas, norm=norm, cmap=cmap)
    if i > 0:
        ax.set_yticklabels([])

# === Bottom row (3D surface plots) ===
for i, (ax, (letter, model_name, model_path), data) in enumerate(zip(bot_axes, models, datas)):
    fade = 0.75 if letter == "S" else None
    surface(ax, model_name, model_path, data, last_alpha_betas, norm=norm, cmap=cmap, fade_beyond=fade)
    if i == 0:
        ax.zaxis.set_rotate_label(False)
        ax.zaxis.label.set_rotation(90)
        ax.zaxis._axinfo['juggled'] = (1, 2, 0)
    else:
        ax.set_zticklabels([])

# === Single shared colorbar ===
cbar_ax = fig.add_axes([0.955, 0.27, 0.012, 0.43])
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, cax=cbar_ax, label=r'Loss ($\gamma$-scaled; $\gamma = 0.6$)')
cbar_ax.yaxis.label.set_size(14)

# === Shared axis labels ===
fig.text(0.5, 0.035, r'$\alpha : a_{GT} \to \hat{a}_\mathrm{GBP-Pretrained}$',
         ha='center', va='center', fontsize=16)
fig.text(0.015, 0.5, r'$\beta : a_{GT} \to \hat{a}_\mathrm{GBP-Adversarial}$',
         ha='center', va='center', rotation='vertical', fontsize=16)

# === Tighter layout ===
plt.subplots_adjust(
    left=0.05,
    right=0.94,
    top=0.92,
    bottom=0.07,
    wspace=0.05,
    hspace=0.1
)

plt.savefig("loss_landscape_2x4.png", dpi=500, bbox_inches='tight')
plt.savefig("loss_landscape_2x4.pgf", bbox_inches='tight')
plt.show()
