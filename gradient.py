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
    ("T", "DINO-WM", "pusht.pretrained"),
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
cmap = cm.viridis  # unified colormap for both contour & 3D

TICKS = [-1, -0.5, 0, 0.5, 1]

# === Plot functions ===
def contour(ax, model_name, model_path, data, last_alpha_beta, norm, cmap):
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

def surface(ax, model_name, model_path, data, last_alpha_beta, norm, cmap):
    A, B, Z, *_ = data
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

# === Figure + layout ===
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0], projection='3d'),
    fig.add_subplot(gs[1, 1], projection='3d')
]

# === Top row (2D contour plots) ===
for ax, (_, model_name, model_path), data in zip(axes[:2], models, datas):
    contour(ax, model_name, model_path, data, last_alpha_betas, norm=norm, cmap=cmap)

# === Bottom row (3D surface plots) ===
for ax, (_, model_name, model_path), data in zip(axes[2:], models, datas):
    surface(ax, model_name, model_path, data, last_alpha_betas, norm=norm, cmap=cmap)

# === Single shared colorbar ===
cbar_ax = fig.add_axes([0.945, 0.49-.215, 0.018, 0.43])
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, cax=cbar_ax, label=r'Loss ($\gamma$-scaled; $\gamma = 0.6$)')
cbar_ax.yaxis.label.set_size(14)

# === Shared axis labels ===
fig.text(0.5, 0.035, r'$\alpha : a_{GT} \to \hat{a}_\mathrm{GBP-Pretrained}$',
         ha='center', va='center', fontsize=16)
fig.text(0.035, 0.5, r'$\beta : a_{GT} \to \hat{a}_\mathrm{GBP-Adversarial}$',
         ha='center', va='center', rotation='vertical', fontsize=16)

# === Tighter layout ===
plt.subplots_adjust(
    left=0.08,   # more room for y-label
    right=0.93,  # colorbar sits just outside
    top=0.92,
    bottom=0.07,
    wspace=0.00,  # reduced horizontal spacing
    hspace=0.1    # reduced vertical spacing
)

plt.savefig("loss_landscape_2x2_compact.png", dpi=500, bbox_inches='tight')
plt.savefig('loss_landscape_2x2_compact.pgf', bbox_inches='tight')
plt.show()
