"""Plotting for Experiment 2: MLM results (NL and protein)."""

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


def _pretty_condition(positional_mode: str, drop_positions_tokens: int) -> str:
    if positional_mode == "none":
        return "None"
    if positional_mode == "learned_abs" and int(drop_positions_tokens) < 0:
        return "Absolute"
    if positional_mode == "learned_abs" and int(drop_positions_tokens) >= 0:
        return "DroPE"
    if positional_mode == "rotary" and int(drop_positions_tokens) < 0:
        return "RoPE"
    if positional_mode == "rotary" and int(drop_positions_tokens) >= 0:
        return "RoPE off"
    return f"{positional_mode} (drop={int(drop_positions_tokens)})"


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
        _pretty_condition(pm, dt)
        for pm, dt in zip(history_df["positional_mode"], history_df["drop_positions_tokens"])
    ]
    history_df["Variant"] = history_df["Attention"] + " / " + history_df["Condition"]

    # Figure 1: Training loss over time
    train_df = history_df[history_df["split"] == "train"].copy()
    if len(train_df) > 0:
        g = sns.relplot(
            data=train_df, x="tokens_seen", y="loss", hue="Variant",
            col="phase", kind="line", errorbar="sd", height=3.2, aspect=1.2,
        )
        g.set_axis_labels("Tokens seen", "Train loss")
        for ax in g.axes.flatten():
            sns.despine(ax=ax)
        _savefig(g.figure, out_dir, "figure1_train_loss_over_time")
        plt.close(g.figure)

    # Figure 2: Validation-short metrics over time
    valid_short_df = history_df[history_df["split"] == "valid_short"].copy()
    if len(valid_short_df) > 0:
        metric_cols = ["loss", "acc", "f1", "mcc"]
        for col in metric_cols:
            if col not in valid_short_df.columns:
                valid_short_df[col] = np.nan
        long_df = valid_short_df.melt(
            id_vars=["phase", "tokens_seen", "Variant"],
            value_vars=metric_cols, var_name="metric", value_name="value",
        )
        g = sns.relplot(
            data=long_df, x="tokens_seen", y="value", hue="Variant",
            row="phase", col="metric", kind="line", errorbar="sd", height=2.6, aspect=1.0,
        )
        g.set_axis_labels("Tokens seen", "Value")
        for ax in g.axes.flatten():
            sns.despine(ax=ax)
        _savefig(g.figure, out_dir, "figure2a_valid_short_metrics_over_time")
        plt.close(g.figure)

    # Figure 2b: Validation-long metrics over time
    valid_long_df = history_df[history_df["split"] == "valid_long"].copy()
    if len(valid_long_df) > 0:
        metric_cols = ["loss", "acc", "f1", "mcc"]
        for col in metric_cols:
            if col not in valid_long_df.columns:
                valid_long_df[col] = np.nan
        long_df = valid_long_df.melt(
            id_vars=["phase", "tokens_seen", "Variant"],
            value_vars=metric_cols, var_name="metric", value_name="value",
        )
        g = sns.relplot(
            data=long_df, x="tokens_seen", y="value", hue="Variant",
            row="phase", col="metric", kind="line", errorbar="sd", height=2.6, aspect=1.0,
        )
        g.set_axis_labels("Tokens seen", "Value")
        for ax in g.axes.flatten():
            sns.despine(ax=ax)
        _savefig(g.figure, out_dir, "figure2b_valid_long_metrics_over_time")
        plt.close(g.figure)

    # Figure 3: Final validation metrics (best eval, short and long)
    if len(results_df) > 0:
        results_df["Attention"] = results_df["attention_type"].map(_pretty_attention)
        results_df["Condition"] = [
            _pretty_condition(pm, dt)
            for pm, dt in zip(results_df["positional_mode"], results_df["drop_positions_tokens"])
        ]
        results_df["Variant"] = results_df["Attention"] + " / " + results_df["Condition"]

        valid_rows = []

        def _maybe_add_valid(row: pd.Series, *, dataset: str, prefix: str, split: str) -> None:
            metric_map = {
                "loss": f"{prefix}_best_valid_{split}_loss",
                "acc": f"{prefix}_best_valid_{split}_acc",
                "f1": f"{prefix}_best_valid_{split}_f1",
                "mcc": f"{prefix}_best_valid_{split}_mcc",
            }
            for metric, col in metric_map.items():
                if col in results_df.columns:
                    valid_rows.append({
                        "Variant": row["Variant"],
                        "dataset": dataset,
                        "split": split,
                        "metric": metric,
                        "value": row[col],
                    })

        for _, row in results_df.iterrows():
            for split in ("short", "long"):
                _maybe_add_valid(row, dataset="NL", prefix="nl", split=split)
                _maybe_add_valid(row, dataset="Protein", prefix="prot", split=split)

        if len(valid_rows) > 0:
            valid_plot_df = pd.DataFrame(valid_rows)
            g = sns.catplot(
                data=valid_plot_df, x="Variant", y="value", hue="split",
                col="dataset", row="metric",
                kind="bar", height=2.4, aspect=1.6, errorbar="sd",
            )
            g.set_axis_labels("", "Value")
            for ax in g.axes.flatten():
                ax.tick_params(axis="x", rotation=45)
                sns.despine(ax=ax)
            _savefig(g.figure, out_dir, "figure3_final_valid_metrics")
            plt.close(g.figure)

    # Figure 4: Final test performance (extended context)
    if len(results_df) > 0:
        test_rows = []

        def _maybe_add_test(row: pd.Series, *, dataset: str, prefix: str) -> None:
            metric_map = {
                "loss": f"{prefix}_test_loss",
                "acc": f"{prefix}_test_acc",
                "f1": f"{prefix}_test_f1",
                "mcc": f"{prefix}_test_mcc",
            }
            for metric, col in metric_map.items():
                if col in results_df.columns:
                    test_rows.append({
                        "Variant": row["Variant"],
                        "dataset": dataset,
                        "metric": metric,
                        "value": row[col],
                    })

        for _, row in results_df.iterrows():
            _maybe_add_test(row, dataset="NL", prefix="nl")
            _maybe_add_test(row, dataset="Protein", prefix="prot")

        if len(test_rows) > 0:
            test_plot_df = pd.DataFrame(test_rows)
            g = sns.catplot(
                data=test_plot_df, x="Variant", y="value", col="dataset", row="metric",
                kind="bar", height=2.4, aspect=1.6, errorbar="sd",
            )
            g.set_axis_labels("", "Value")
            for ax in g.axes.flatten():
                ax.tick_params(axis="x", rotation=45)
                sns.despine(ax=ax)
            _savefig(g.figure, out_dir, "figure4_final_test_metrics")
            plt.close(g.figure)
