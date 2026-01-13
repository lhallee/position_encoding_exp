from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _style() -> None:
    # Publication-ish defaults (works well for both PNG and PDF)
    sns.set_theme(style="ticks", context="paper", font="DejaVu Sans")
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "regular"
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    sns.set_palette("colorblind")


def _pretty_attention(attn: str) -> str:
    if attn == "causal":
        return "Causal (unidirectional)"
    if attn == "bidirectional":
        return "Bidirectional"
    return attn


def _pretty_condition(positional_mode: str, drop_positions_step: int) -> str:
    if positional_mode == "none":
        return "No positional embeddings"
    if positional_mode == "learned_abs" and int(drop_positions_step) < 0:
        return "Learned absolute positions"
    if positional_mode == "learned_abs" and int(drop_positions_step) >= 0:
        return "Drop positions after eval #2 (DroPE-like)"
    return f"{positional_mode} (drop={int(drop_positions_step)})"


def _savefig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png")


def plot_all(*, results_csv: Path, out_dir: Path) -> None:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    # Aggregate over seeds: mean and std for error reporting.
    gdf = (
        df.groupby(
            ["attention_type", "positional_mode", "drop_positions_step", "label_mode", "n_layers", "d_model"],
            as_index=False,
        )["eval_acc"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "eval_acc_mean", "std": "eval_acc_std"})
    )

    gdf["Attention"] = gdf["attention_type"].map(_pretty_attention)
    gdf["Condition"] = [
        _pretty_condition(pm, ds) for pm, ds in zip(gdf["positional_mode"], gdf["drop_positions_step"])
    ]
    gdf["Label mode"] = gdf["label_mode"].map({"true": "True labels", "random": "Random labels (control)"})
    gdf["Number of layers"] = gdf["n_layers"]
    gdf["Hidden size"] = gdf["d_model"]

    chance = 1.0 / float(df["seq_len"].iloc[0])

    # ----------------------------
    # Figure 1: Heatmap grid (true labels)
    # ----------------------------
    hdf = gdf[gdf["label_mode"] == "true"].copy()
    conditions = [
        "No positional embeddings",
        "Learned absolute positions",
        "Drop positions after eval #2 (DroPE-like)",
    ]
    attention_order = ["Bidirectional", "Causal (unidirectional)"]

    fig, axes = plt.subplots(
        nrows=len(attention_order),
        ncols=len(conditions),
        figsize=(10.8, 6.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # Use chance as the lower bound for better contrast while keeping a shared scale.
    vmin, vmax = chance, 1.0
    last_hm = None
    for r, attn in enumerate(attention_order):
        for c, cond in enumerate(conditions):
            ax = axes[r, c]
            sdf = hdf[(hdf["Attention"] == attn) & (hdf["Condition"] == cond)]
            pivot = sdf.pivot(index="Number of layers", columns="Hidden size", values="eval_acc_mean")

            last_hm = sns.heatmap(
                pivot,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
                cbar=False,
                linewidths=0.6,
                linecolor="white",
                square=True,
            )
            if c == 0:
                ax.set_ylabel(f"{attn}\n\nNumber of layers")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Hidden size" if r == len(attention_order) - 1 else "")
            ax.tick_params(axis="x", rotation=0)
            ax.tick_params(axis="y", rotation=0)
            sns.despine(ax=ax, left=False, bottom=False)

    # Shared colorbar
    if last_hm is not None:
        cbar = fig.colorbar(last_hm.collections[0], ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label("Accuracy (mean over seeds)")

    _savefig(fig, out_dir, "figure1_heatmap_grid")
    plt.close(fig)

    # ----------------------------
    # Figure 2: Control (random labels) vs chance
    # ----------------------------
    cdf_raw = df[df["label_mode"] == "random"].copy()
    if len(cdf_raw) > 0:
        fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
        cdf_raw["Attention"] = cdf_raw["attention_type"].map(_pretty_attention)
        sns.barplot(data=cdf_raw, x="Attention", y="eval_acc", ax=ax, errorbar=("sd", 1), capsize=0.15)
        ax.axhline(chance, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy")
        ymax = max(chance * 5.0, float(cdf_raw["eval_acc"].max()) * 1.35)
        ax.set_ylim(0.0, min(1.0, ymax))
        sns.despine(ax=ax)
        _savefig(fig, out_dir, "figure2_control_random_labels")
        plt.close(fig)

    # ----------------------------
    # Figure 3: Summary bars at the largest model (true labels)
    # ----------------------------
    tdf_raw = df[df["label_mode"] == "true"].copy()
    if len(tdf_raw) > 0:
        max_layers = int(tdf_raw["n_layers"].max())
        max_width = int(tdf_raw["d_model"].max())
        sdf_raw = tdf_raw[(tdf_raw["n_layers"] == max_layers) & (tdf_raw["d_model"] == max_width)].copy()
        if len(sdf_raw) > 0:
            sdf_raw["Attention"] = sdf_raw["attention_type"].map(_pretty_attention)
            sdf_raw["Condition"] = [
                _pretty_condition(pm, ds) for pm, ds in zip(sdf_raw["positional_mode"], sdf_raw["drop_positions_step"])
            ]
            hue_order = [
                "No positional embeddings",
                "Drop positions after eval #2 (DroPE-like)",
                "Learned absolute positions",
            ]
            fig, ax = plt.subplots(figsize=(8.6, 3.8), constrained_layout=True)
            sns.barplot(
                data=sdf_raw,
                x="Attention",
                y="eval_acc",
                hue="Condition",
                ax=ax,
                errorbar=("sd", 1),
                capsize=0.12,
                hue_order=hue_order,
            )
            ax.set_xlabel("")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.0, 1.0)
            ax.legend(title="", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
            sns.despine(ax=ax)
            _savefig(fig, out_dir, "figure3_largest_model_comparison")
            plt.close(fig)

    # ----------------------------
    # Figure 4: Key hypothesis view (no positional embeddings)
    # ----------------------------
    ndf = df[(df["label_mode"] == "true") & (df["positional_mode"] == "none")].copy()
    if len(ndf) > 0:
        ndf["Attention"] = ndf["attention_type"].map(_pretty_attention)
        ndf["Number of layers"] = ndf["n_layers"]
        ndf["Hidden size"] = ndf["d_model"]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10.2, 3.8), sharey=True, constrained_layout=False)
        for ax, attn in zip(axes, attention_order):
            sdf = ndf[ndf["Attention"] == attn]
            sns.lineplot(
                data=sdf,
                x="Hidden size",
                y="eval_acc",
                hue="Number of layers",
                marker="o",
                ax=ax,
                errorbar=("sd", 1),
                palette="viridis",
            )
            ax.axhline(chance, color="black", linestyle="--", linewidth=1.0)
            ax.set_xlabel("Hidden size")
            ax.set_ylabel("Accuracy" if attn == attention_order[0] else "")
            ax.set_ylim(0.0, 1.0)
            ax.legend(title="Number of layers", frameon=False, loc="lower right")
            sns.despine(ax=ax)

        fig.tight_layout()
        _savefig(fig, out_dir, "figure4_no_positional_embeddings_key_view")
        plt.close(fig)
