import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .figures import GRID, INK, INK2, S1, S2, S3, SURFACE

SERIES = {"beverage_can": (S1, "o", "Cans"), "glass_jar": (S2, "s", "Jars"), "mug": (S3, "D", "Mugs")}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def detection_chart(sweeps, out_png):
    groups = {}
    for s in sweeps:
        if s["eligible"]:
            groups.setdefault(s["sweep"], []).append((s["magnitude"], s["detected"] / s["eligible"]))
    labels = {"scale": ("Scale error, %", lambda m: f"{(m - 1) * 100:.0f}"), "com_offset_both": ("COM offset, mm", lambda m: f"{m * 1000:g}"), "mass_both": ("Mass error, %", lambda m: f"{(m - 1) * 100:.0f}"), "inertia_scale_both": ("Inertia error, %", lambda m: f"{(m - 1) * 100:.0f}"), "collider_offset_mjcf": ("MJCF collider offset, mm", lambda m: f"{m * 1000:g}")}
    keys = [k for k in labels if k in groups]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.1 * len(keys), 2.6), facecolor=SURFACE, sharey=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        _style(ax)
        pts = sorted(groups[k])
        xs = np.arange(len(pts))
        ax.plot(xs, [p[1] * 100 for p in pts], color=S1, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter(xs, [p[1] * 100 for p in pts], s=36, color=S1, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.set_xticks(xs, [labels[k][1](p[0]) for p in pts], rotation=0, fontsize=8.5)
        ax.set_title(labels[k][0], fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(-5, 105)
    np.atleast_1d(axes)[0].set_ylabel("Detected, % of assets", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return out_png
