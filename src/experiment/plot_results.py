import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path


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
        return "Causal"
    if attn == "bidirectional":
        return "Bidirectional"
    if attn == "dual_triangle":
        return "DualTriangle"
    return attn


def _pretty_condition(positional_mode: str, drop_positions_step: int) -> str:
    if positional_mode == "none":
        return "None"
    if positional_mode == "learned_abs" and int(drop_positions_step) < 0:
        return "Absolute"
    if positional_mode == "learned_abs" and int(drop_positions_step) >= 0:
        return "DroPE"
    if positional_mode == "rotary" and int(drop_positions_step) < 0:
        return "RoPE"
    if positional_mode == "rotary" and int(drop_positions_step) >= 0:
        return "RoPE off"
    return f"{positional_mode} (drop={int(drop_positions_step)})"


def _group_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Compute mean/std/n and a normal-approx 95% CI half-width for eval_acc.

    Note: with small #seeds, this is an approximation (z=1.96).
    """
    g = df.groupby(group_cols, as_index=False)["eval_acc"].agg(["mean", "std", "count"]).reset_index()
    g = g.rename(columns={"mean": "acc_mean", "std": "acc_std", "count": "n"})
    g["acc_std"] = g["acc_std"].fillna(0.0)
    g["acc_sem"] = g["acc_std"] / np.sqrt(g["n"].clip(lower=1))
    g["acc_ci95"] = 1.96 * g["acc_sem"]
    return g


def _savefig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png")


def plot_all(*, results_csv: Path, out_dir: Path) -> None:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    # Aggregate over seeds: mean and std for error reporting.
    # Note: `eval_acc` is the LAST eval accuracy from training (not best), by design.
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
    gdf["Layers"] = gdf["n_layers"]
    gdf["Hidden Size"] = gdf["d_model"]

    chance = 1.0 / float(df["seq_len"].iloc[0])

    # ----------------------------
    # Figure 1: Heatmap grid (true labels)
    # ----------------------------
    hdf = gdf[gdf["label_mode"] == "true"].copy()
    conditions = ["None", "Absolute", "DroPE", "RoPE", "RoPE off"]
    attention_order = ["Bidirectional", "DualTriangle", "Causal"]

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
            if len(sdf) == 0:
                ax.set_title(cond if r == 0 else "")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                sns.despine(ax=ax, left=True, bottom=True)
                continue

            pivot = sdf.pivot(index="Layers", columns="Hidden Size", values="eval_acc_mean")

            last_hm = sns.heatmap(
                pivot,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
                cbar=False,
                annot=True,
                fmt=".2f",
                annot_kws={"fontsize": 8},
                linewidths=0.6,
                linecolor="white",
                square=True,
            )
            if c == 0:
                ax.set_ylabel(f"{attn}\n\nLayers")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Hidden Size" if r == len(attention_order) - 1 else "")
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
        stats = _group_stats(cdf_raw, ["Attention"]).sort_values("Attention")
        x = np.arange(len(stats))
        ax.bar(x, stats["acc_mean"], color=sns.color_palette("colorblind", n_colors=1)[0])
        ax.errorbar(
            x,
            stats["acc_mean"],
            yerr=stats["acc_ci95"],
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
        )
        ax.set_xticks(x, stats["Attention"].tolist())
        ax.axhline(chance, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy")
        ymax = max(chance * 5.0, float((stats["acc_mean"] + stats["acc_ci95"]).max()) * 1.25)
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
            hue_order = ["None", "DroPE", "Absolute"]
            fig, ax = plt.subplots(figsize=(8.6, 3.8), constrained_layout=True)
            stats = _group_stats(sdf_raw, ["Attention", "Condition"])
            stats["Attention"] = pd.Categorical(stats["Attention"], categories=attention_order, ordered=True)
            stats["Condition"] = pd.Categorical(stats["Condition"], categories=hue_order, ordered=True)
            stats = stats.sort_values(["Attention", "Condition"])

            attn_vals = attention_order
            cond_vals = hue_order
            n_attn = len(attn_vals)
            n_cond = len(cond_vals)
            width = 0.22
            base_x = np.arange(n_attn)

            palette = sns.color_palette("colorblind", n_colors=n_cond)
            for j, cond in enumerate(cond_vals):
                sub = stats[stats["Condition"] == cond]
                sub = sub.set_index("Attention").reindex(attn_vals).reset_index()
                xs = base_x + (j - (n_cond - 1) / 2) * width
                ax.bar(xs, sub["acc_mean"], width=width, label=cond, color=palette[j])
                ax.errorbar(
                    xs,
                    sub["acc_mean"],
                    yerr=sub["acc_ci95"],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=3,
                    capthick=1.0,
                )

            ax.set_xticks(base_x, attn_vals)
            ax.set_xlabel("")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.0, 1.0)
            ax.legend(title="Positional Encoding", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
            sns.despine(ax=ax)
            _savefig(fig, out_dir, "figure3_largest_model_comparison")
            plt.close(fig)

    # ----------------------------
    # Figure 4: Key hypothesis view (no positional embeddings)
    # ----------------------------
    ndf = df[(df["label_mode"] == "true") & (df["positional_mode"] == "none")].copy()
    if len(ndf) > 0:
        ndf["Attention"] = ndf["attention_type"].map(_pretty_attention)
        ndf["Layers"] = ndf["n_layers"]
        ndf["Hidden Size"] = ndf["d_model"]

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13.2, 3.8), sharey=True, constrained_layout=True)
        layer_vals = sorted(ndf["Layers"].unique())
        layer_colors = sns.color_palette("viridis", n_colors=len(layer_vals))

        for ax, attn in zip(axes, attention_order):
            ax.set_title(attn)
            sdf = ndf[ndf["Attention"] == attn]
            stats = _group_stats(sdf, ["Layers", "Hidden Size"]).sort_values(["Layers", "Hidden Size"])
            for color, layer in zip(layer_colors, layer_vals):
                sub = stats[stats["Layers"] == layer]
                ax.plot(sub["Hidden Size"], sub["acc_mean"], marker="o", color=color, linewidth=1.8, label=str(layer))
                ax.errorbar(
                    sub["Hidden Size"],
                    sub["acc_mean"],
                    yerr=sub["acc_ci95"],
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=3,
                    capthick=1.0,
                )
            ax.axhline(chance, color="black", linestyle="--", linewidth=1.0)
            ax.set_xlabel("Hidden Size")
            ax.set_ylabel("Accuracy" if attn == attention_order[0] else "")
            ax.set_ylim(0.0, 1.0)
            ax.legend(title="Layers", frameon=False, loc="lower right")
            sns.despine(ax=ax)

        _savefig(fig, out_dir, "figure4_no_positional_embeddings_key_view")
        plt.close(fig)
