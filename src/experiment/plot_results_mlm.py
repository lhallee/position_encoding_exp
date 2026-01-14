import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path


def _style() -> None:
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


def _savefig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png")


def plot_all(*, results_csv: Path, history_csv: Path, out_dir: Path) -> None:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(results_csv)
    history_df = pd.read_csv(history_csv)

    history_df["Attention"] = history_df["attention_type"].map(_pretty_attention)
    history_df["Condition"] = [
        _pretty_condition(pm, ds)
        for pm, ds in zip(history_df["positional_mode"], history_df["drop_positions_step"])
    ]
    history_df["Variant"] = history_df["Attention"] + " / " + history_df["Condition"]

    # ----------------------------
    # Figure 1: Training loss over time
    # ----------------------------
    train_df = history_df[history_df["split"] == "train"].copy()
    if len(train_df) > 0:
        g = sns.relplot(
            data=train_df,
            x="global_step",
            y="loss",
            hue="Variant",
            col="phase",
            kind="line",
            errorbar="sd",
            height=3.2,
            aspect=1.2,
        )
        g.set_axis_labels("Global step", "Train loss")
        for ax in g.axes.flatten():
            sns.despine(ax=ax)
        _savefig(g.figure, out_dir, "figure1_train_loss_over_time")
        plt.close(g.figure)

    # ----------------------------
    # Figure 2: Validation metrics over time
    # ----------------------------
    valid_df = history_df[history_df["split"] == "valid"].copy()
    if len(valid_df) > 0:
        metric_cols = ["loss", "acc", "f1", "mcc"]
        for col in metric_cols:
            if col not in valid_df.columns:
                valid_df[col] = np.nan
        long_df = valid_df.melt(
            id_vars=["phase", "global_step", "Variant"],
            value_vars=metric_cols,
            var_name="metric",
            value_name="value",
        )
        g = sns.relplot(
            data=long_df,
            x="global_step",
            y="value",
            hue="Variant",
            row="phase",
            col="metric",
            kind="line",
            errorbar="sd",
            height=2.6,
            aspect=1.0,
        )
        g.set_axis_labels("Global step", "Value")
        for ax in g.axes.flatten():
            sns.despine(ax=ax)
        _savefig(g.figure, out_dir, "figure2_valid_metrics_over_time")
        plt.close(g.figure)

    # ----------------------------
    # Figure 3: Final test performance (extended context)
    # ----------------------------
    if len(results_df) > 0:
        results_df["Attention"] = results_df["attention_type"].map(_pretty_attention)
        results_df["Condition"] = [
            _pretty_condition(pm, ds)
            for pm, ds in zip(results_df["positional_mode"], results_df["drop_positions_step"])
        ]
        results_df["Variant"] = results_df["Attention"] + " / " + results_df["Condition"]

        rows = []
        for _, row in results_df.iterrows():
            rows.append(
                {
                    "Variant": row["Variant"],
                    "dataset": "NL",
                    "metric": "loss",
                    "value": row["nl_test_loss"],
                }
            )
            rows.append(
                {
                    "Variant": row["Variant"],
                    "dataset": "NL",
                    "metric": "acc",
                    "value": row["nl_test_acc"],
                }
            )
            rows.append(
                {
                    "Variant": row["Variant"],
                    "dataset": "NL",
                    "metric": "f1",
                    "value": row["nl_test_f1"],
                }
            )
            rows.append(
                {
                    "Variant": row["Variant"],
                    "dataset": "NL",
                    "metric": "mcc",
                    "value": row["nl_test_mcc"],
                }
            )
            rows.append(
                {
                    "Variant": row["Variant"],
                    "dataset": "Protein",
                    "metric": "loss",
                    "value": row["prot_test_loss"],
                }
            )
            rows.append(
                {
                    "Variant": row["Variant"],
                    "dataset": "Protein",
                    "metric": "acc",
                    "value": row["prot_test_acc"],
                }
            )
            rows.append(
                {
                    "Variant": row["Variant"],
                    "dataset": "Protein",
                    "metric": "f1",
                    "value": row["prot_test_f1"],
                }
            )
            rows.append(
                {
                    "Variant": row["Variant"],
                    "dataset": "Protein",
                    "metric": "mcc",
                    "value": row["prot_test_mcc"],
                }
            )

        test_df = pd.DataFrame(rows)
        g = sns.catplot(
            data=test_df,
            x="Variant",
            y="value",
            col="dataset",
            row="metric",
            kind="bar",
            height=2.4,
            aspect=1.6,
            errorbar="sd",
        )
        g.set_axis_labels("", "Value")
        for ax in g.axes.flatten():
            ax.tick_params(axis="x", rotation=45)
            sns.despine(ax=ax)
        _savefig(g.figure, out_dir, "figure3_extended_context_test")
        plt.close(g.figure)
