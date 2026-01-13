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
            ["attention_type", "positional_mode", "drop_positions_step", "n_layers", "d_model"],
            as_index=False,
        )["eval_acc"]
        .mean()
        .rename(columns={"eval_acc": "eval_acc_mean"})
    )

    # Heatmaps: for each attention_type + positional condition, accuracy vs layers/d_model
    for attention_type in sorted(gdf["attention_type"].unique()):
        adf = gdf[gdf["attention_type"] == attention_type]
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
    sdf = gdf[gdf["drop_positions_step"] == -1]
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

