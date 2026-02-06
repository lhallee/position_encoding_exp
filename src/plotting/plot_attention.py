"""Visualization of attention patterns: bidirectional, causal, dual triangle."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def _style() -> None:
    sns.set_theme(style="ticks", context="paper", font="DejaVu Sans")
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    sns.set_palette("colorblind")


def _matrix(ax: plt.Axes, data: np.ndarray, *, title: str) -> None:
    ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def _dual_triangle_matrix(n: int) -> np.ndarray:
    lower = np.full((n, n), 0.75)
    upper = np.full((n, n), 0.35)
    tri = lower.copy()
    tri[np.triu_indices(n, k=1)] = upper[np.triu_indices(n, k=1)]
    return tri


def _causal_matrix(n: int) -> np.ndarray:
    mat = np.zeros((n, n))
    mat[np.tril_indices(n, k=0)] = 0.75
    return mat


def _bidirectional_matrix(n: int) -> np.ndarray:
    return np.full((n, n), 0.75)


def plot_dual_triangle_attention(*, out_dir: Path, n: int = 9) -> None:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.4, 3.5), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1.05, 1.05, 1.4])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    _matrix(ax0, _bidirectional_matrix(n), title="Bidirectional")
    _matrix(ax1, _causal_matrix(n), title="Causal (lower triangle)")
    _matrix(ax2, _dual_triangle_matrix(n), title="Dual Triangle")

    ax2.annotate(
        "Upper triangle (j > i)\nHeads N..2N-1",
        xy=(0.78, 0.22), xycoords="axes fraction", ha="center", va="center",
        fontsize=8, color="black",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )
    ax2.annotate(
        "Lower triangle (j <= i)\nHeads 0..N-1",
        xy=(0.26, 0.78), xycoords="axes fraction", ha="center", va="center",
        fontsize=8, color="black",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )

    for ax in (ax0, ax1, ax2):
        ax.set_xlabel("Key position (j)")
    ax0.set_ylabel("Query position (i)")

    fig.suptitle("Dual Triangle Attention: head-routed block mask via flex_attention", y=1.03)
    fig.savefig(out_dir / "dual_triangle_attention.png")
    plt.close(fig)


def _add_box(
    ax: plt.Axes, xy: tuple[float, float], text: str,
    *, w: float = 0.22, h: float = 0.12, fc: str = "white",
) -> tuple[float, float, float, float]:
    x, y = xy
    rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc, edgecolor="0.4", linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=8)
    return (x - w / 2, y - h / 2, w, h)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.0, color="0.25"),
    )


def _flow_axes(fig: plt.Figure, pos: tuple[float, float, float, float], title: str) -> plt.Axes:
    ax = fig.add_axes(pos)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title(title, fontsize=10, weight="bold", pad=6)
    return ax


def plot_attention_flow(*, out_dir: Path) -> None:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.8, 6.2), constrained_layout=False)
    left = 0.05
    width = 0.28
    gap = 0.045
    bottom = 0.10
    height = 0.82

    ax_bi = _flow_axes(fig, (left, bottom, width, height), "Bidirectional (Standard)")
    ax_causal = _flow_axes(fig, (left + width + gap, bottom, width, height), "Causal (Unidirectional)")
    ax_dual = _flow_axes(fig, (left + 2 * (width + gap), bottom, width, height), "Dual Triangle (Flex Attention)")

    y = {
        "x": 0.92, "qkv": 0.78, "split": 0.63,
        "attn": 0.42, "softmax": 0.28, "out": 0.12,
    }

    # Bidirectional
    _add_box(ax_bi, (0.5, y["x"]), "Input x")
    _add_box(ax_bi, (0.5, y["qkv"]), "Linear: QKV")
    _add_box(ax_bi, (0.5, y["split"]), "Split -> q, k, v")
    _add_box(ax_bi, (0.5, y["attn"]), "flex_attention\n(full mask)")
    _add_box(ax_bi, (0.5, y["out"]), "Output")
    _arrow(ax_bi, (0.5, y["x"] - 0.06), (0.5, y["qkv"] + 0.06))
    _arrow(ax_bi, (0.5, y["qkv"] - 0.06), (0.5, y["split"] + 0.06))
    _arrow(ax_bi, (0.5, y["split"] - 0.06), (0.5, y["attn"] + 0.06))
    _arrow(ax_bi, (0.5, y["attn"] - 0.07), (0.5, y["out"] + 0.06))
    ax_bi.text(0.5, 0.52, "Full matrix\n(all i,j)", ha="center", va="center", fontsize=7, color="0.35")

    # Causal
    _add_box(ax_causal, (0.5, y["x"]), "Input x")
    _add_box(ax_causal, (0.5, y["qkv"]), "Linear: QKV")
    _add_box(ax_causal, (0.5, y["split"]), "Split -> q, k, v")
    _add_box(ax_causal, (0.5, y["attn"]), "flex_attention\n(causal mask)")
    _add_box(ax_causal, (0.5, y["out"]), "Output")
    _arrow(ax_causal, (0.5, y["x"] - 0.06), (0.5, y["qkv"] + 0.06))
    _arrow(ax_causal, (0.5, y["qkv"] - 0.06), (0.5, y["split"] + 0.06))
    _arrow(ax_causal, (0.5, y["split"] - 0.06), (0.5, y["attn"] + 0.06))
    _arrow(ax_causal, (0.5, y["attn"] - 0.07), (0.5, y["out"] + 0.06))
    ax_causal.text(0.5, 0.52, "Lower triangle\n(masked future)", ha="center", va="center", fontsize=7, color="0.35")

    # Dual Triangle
    _add_box(ax_dual, (0.5, y["x"]), "Input x")
    _add_box(ax_dual, (0.5, y["qkv"]), "Linear: QKV")
    _add_box(ax_dual, (0.5, y["split"]), "Split -> q, k, v")
    _add_box(ax_dual, (0.5, 0.52), "Reshape to\n2N heads x d/2", w=0.33)
    _arrow(ax_dual, (0.5, y["split"] - 0.06), (0.5, 0.58))

    _add_box(ax_dual, (0.5, y["attn"]), "flex_attention\n(head-routed mask)", w=0.40)
    _arrow(ax_dual, (0.5, 0.46), (0.5, y["attn"] + 0.06))

    ax_dual.text(0.5, 0.34, "Heads 0..N-1: lower tri\nHeads N..2N-1: upper tri",
                 ha="center", va="center", fontsize=7, color="0.35")

    _add_box(ax_dual, (0.5, y["out"]), "Reshape back\n+ output proj", w=0.33)
    _arrow(ax_dual, (0.5, y["attn"] - 0.07), (0.5, y["out"] + 0.06))

    fig.suptitle("Attention flow: standard vs causal vs dual triangle (all via flex_attention)", y=0.98)
    fig.savefig(out_dir / "attention_flow_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    out = Path("figures")
    plot_dual_triangle_attention(out_dir=out)
    plot_attention_flow(out_dir=out)
