from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def plot_all(*, results_csv: Path, out_dir: Path) -> None:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    # aggregate over seeds
    gdf = (
        df.groupby(
            ["attention_type", "positional_mode", "drop_positions_step", "label_mode", "n_layers", "d_model"],
            as_index=False,
        )["eval_acc"]
        .mean()
        .rename(columns={"eval_acc": "eval_acc_mean"})
    )

    # Heatmaps: for each attention_type + positional condition, accuracy vs layers/d_model
    for attention_type in sorted(gdf["attention_type"].unique()):
        adf = gdf[gdf["attention_type"] == attention_type]
        adf = adf[adf["label_mode"] == "true"]
        for (positional_mode, drop_step), sdf in adf.groupby(["positional_mode", "drop_positions_step"]):
            pivot = sdf.pivot(index="n_layers", columns="d_model", values="eval_acc_mean")

            plt.figure(figsize=(7, 4.5))
            sns.heatmap(
                pivot,
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                annot=True,
                fmt=".2f",
                cbar_kws={"label": "Accuracy"},
            )
            title = f"{attention_type} | pos={positional_mode}"
            if int(drop_step) >= 0:
                title += f" | drop@{int(drop_step)}"
            plt.title(title)
            plt.xlabel("d_model")
            plt.ylabel("n_layers")

            safe = f"{attention_type}__{positional_mode}__drop{int(drop_step)}.png"
            plt.savefig(out_dir / safe)
            plt.close()

    # Lineplot: accuracy vs layers, colored by attention type, faceted by d_model
    plt.figure(figsize=(8, 4.5))
    sdf = gdf[(gdf["drop_positions_step"] == -1) & (gdf["label_mode"] == "true")]
    sns.lineplot(
        data=sdf,
        x="n_layers",
        y="eval_acc_mean",
        hue="attention_type",
        style="positional_mode",
        markers=True,
        dashes=False,
    )
    plt.ylim(0.0, 1.0)
    plt.title("Position probe accuracy vs depth (averaged over d_model)")
    plt.xlabel("n_layers")
    plt.ylabel("Accuracy")
    plt.savefig(out_dir / "lineplot_accuracy_vs_layers.png")
    plt.close()

    # Control plot: random-label runs should be near chance (1/seq_len).
    cdf = gdf[gdf["label_mode"] == "random"]
    if len(cdf) > 0:
        plt.figure(figsize=(7, 4))
        sns.barplot(
            data=cdf,
            x="attention_type",
            y="eval_acc_mean",
            hue="attention_type",
            dodge=False,
        )
        chance = 1.0 / float(df["seq_len"].iloc[0])
        plt.axhline(chance, color="black", linestyle="--", linewidth=1.0, label="chance")
        plt.ylim(0.0, max(0.1, float(cdf["eval_acc_mean"].max()) * 1.2))
        plt.title("Control: random labels (\"shuffled\") on largest model")
        plt.xlabel("attention_type")
        plt.ylabel("Accuracy")
        plt.legend().remove()
        plt.savefig(out_dir / "control_random_labels.png")
        plt.close()
