"""Combined cross-experiment plotting for protein MLM results.

Reads wandb-exported CSVs from results/all/ and per-experiment results.csv
files, then generates journal-ready figure panels comparing attention types
(Dual Triangle, Bidirectional, Causal) across positional encoding strategies
(No PE, RoPE, RoPE-off / DroPE).

Usage:
    py -m src.plotting.plot_combined
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import ttest_ind


# ====================================================================
# Constants
# ====================================================================
DROP_STEP = 17_500  # ~700 M tokens / (~40 k tokens per step)

ATTN_PRETTY = {
    "dual_triangle": "Dual Triangle",
    "causal": "Causal",
    "bidirectional": "Bidirectional",
}
ATTN_ORDER = ["Dual Triangle", "Bidirectional", "Causal"]

COND_ORDER = ["RoPE", "No PE", "RoPE-off"]
COND_LINESTYLES = {"RoPE": "-", "No PE": "--", "RoPE-off": ":"}
COND_LINEWIDTHS = {"RoPE": 1.5, "No PE": 1.3, "RoPE-off": 1.3}

# Seaborn colorblind palette -- fixed indices for consistent colouring
_CB = sns.color_palette("colorblind")
ATTN_COLORS = {
    "Dual Triangle": _CB[0],   # blue
    "Bidirectional": _CB[1],   # orange
    "Causal": _CB[2],          # green
}
COND_COLORS = {
    "RoPE": _CB[0],            # blue
    "RoPE-off": _CB[3],        # red / pink
}

# Column regex for wandb-exported CSV headers
_COL_RE = re.compile(r"^s(\d+)_(.+?)_(rotary|none)_drop(-?\d+) - .+$")


# ====================================================================
# Style
# ====================================================================
def _style():
    """Apply journal-ready matplotlib / seaborn style."""
    sns.set_theme(style="ticks", context="paper", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
    })


def _savefig(fig, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")


# ====================================================================
# Data loading
# ====================================================================
def _condition_label(pos_mode, drop):
    """Map (pos_mode, drop) to a human-readable condition label."""
    drop_str = str(int(float(drop)))
    if str(pos_mode) == "none":
        return "No PE"
    if str(pos_mode) == "rotary" and drop_str == "-1":
        return "RoPE"
    if str(pos_mode) == "rotary" and drop_str == "700000000":
        return "RoPE-off"
    assert False, f"Unknown condition: pos_mode={pos_mode}, drop={drop}"


def load_wandb_csvs(data_dir):
    """Load every CSV in *data_dir* into a long-form DataFrame.

    Returns
    -------
    dict : metric_key (str) -> DataFrame
        Each DataFrame has columns:
        [step, seed, attention, pos_mode, drop, Attention, Condition, value]
    """
    data_dir = Path(data_dir)
    all_data = {}

    for csv_path in sorted(data_dir.glob("*.csv")):
        metric_key = csv_path.stem
        print(f"  Loading {csv_path.name} ...")
        df = pd.read_csv(csv_path)

        records = []
        for col in df.columns:
            if col == "Step" or "__MIN" in col or "__MAX" in col:
                continue

            match = _COL_RE.match(col)
            assert match is not None, f"Failed to parse column: {col}"

            seed = int(match.group(1))
            attention = match.group(2)
            pos_mode = match.group(3)
            drop = match.group(4)

            for step_val, data_val in zip(df["Step"], df[col]):
                if pd.notna(data_val) and str(data_val).strip() != "":
                    records.append({
                        "step": int(float(step_val)),
                        "seed": seed,
                        "attention": attention,
                        "pos_mode": pos_mode,
                        "drop": drop,
                        "value": float(data_val),
                    })

        long_df = pd.DataFrame(records)
        long_df["Attention"] = long_df["attention"].map(ATTN_PRETTY)
        long_df["Condition"] = [
            _condition_label(pm, d)
            for pm, d in zip(long_df["pos_mode"], long_df["drop"])
        ]
        all_data[metric_key] = long_df

    return all_data


def load_final_results(results_dirs):
    """Load and combine ``results.csv`` from multiple experiment directories."""
    frames = []
    for d in results_dirs:
        csv_path = Path(d) / "results.csv"
        assert csv_path.exists(), f"Missing: {csv_path}"
        frames.append(pd.read_csv(csv_path))

    combined = pd.concat(frames, ignore_index=True)
    combined["Attention"] = combined["attention_type"].map(ATTN_PRETTY)
    combined["Condition"] = [
        _condition_label(pm, dt)
        for pm, dt in zip(combined["positional_mode"], combined["drop_positions_tokens"])
    ]
    return combined


# ====================================================================
# Aggregation helpers
# ====================================================================
def _agg_seeds(df, min_seeds=2):
    """Group by (step, Attention, Condition) and compute mean / std."""
    grouped = df.groupby(["step", "Attention", "Condition"])["value"]
    agg = grouped.agg(["mean", "std", "count"]).reset_index()
    agg = agg[agg["count"] >= min_seeds].copy()
    agg["std"] = agg["std"].fillna(0)
    return agg


# ====================================================================
# Significance helpers
# ====================================================================
def _sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _welch_ttest(vals1, vals2):
    """Welch's t-test.  Returns (t, p); (nan, 1.0) when data is insufficient."""
    if len(vals1) < 2 or len(vals2) < 2:
        return np.nan, 1.0
    t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)
    return t_stat, p_val



def _draw_sig_bracket(ax, x1, x2, y, stars, height_frac=0.02):
    """Draw a small bracket between *x1* and *x2* at height *y*."""
    ylim = ax.get_ylim()
    bh = (ylim[1] - ylim[0]) * height_frac
    ax.plot([x1, x1, x2, x2], [y, y + bh, y + bh, y],
            color="black", lw=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + bh * 1.1, stars,
            ha="center", va="bottom", fontsize=7, color="black")


# ====================================================================
# Generic line-plot grid
# ====================================================================
def _plot_line_grid(data_dict, panels, suptitle, out_dir, out_stem,
                    conditions=None, show_drop_line=False, figsize=(11, 8)):
    """Draw a 2x2 grid of mean +/- std line plots.

    Parameters
    ----------
    panels : list of (metric_key, title, ylabel)
    """
    cond_list = conditions if conditions is not None else COND_ORDER
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")

    for ax, (mk, title, ylabel) in zip(axes.flatten(), panels):
        df = data_dict[mk].copy()
        if conditions is not None:
            df = df[df["Condition"].isin(cond_list)]

        agg = _agg_seeds(df)

        for attn in ATTN_ORDER:
            for cond in cond_list:
                sub = agg[(agg["Attention"] == attn) & (agg["Condition"] == cond)]
                if len(sub) == 0:
                    continue
                sub = sub.sort_values("step")
                color = ATTN_COLORS[attn]
                ls = COND_LINESTYLES[cond]
                lw = COND_LINEWIDTHS[cond]
                label = f"{attn} / {cond}"
                ax.plot(sub["step"], sub["mean"], color=color, ls=ls, lw=lw, label=label)
                ax.fill_between(sub["step"],
                                sub["mean"] - sub["std"],
                                sub["mean"] + sub["std"],
                                color=color, alpha=0.12)

        ax.set_title(title)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k")
        )
        sns.despine(ax=ax)

        if show_drop_line:
            ylo, yhi = ax.get_ylim()
            ax.axvline(DROP_STEP, color="gray", ls="--", lw=0.9, alpha=0.7)
            ax.text(DROP_STEP + 300, yhi - (yhi - ylo) * 0.04,
                    "RoPE\ndropped", fontsize=7, color="gray", va="top")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = 3 if len(labels) <= 9 else 2
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, -0.12), frameon=True, fontsize=8)

    _savefig(fig, out_dir, out_stem)
    plt.close(fig)


# ====================================================================
# Generic bar chart
# ====================================================================
def _plot_bar_chart(final_df, metrics, suptitle, out_dir, out_stem,
                    x_var="Condition", hue_var="Attention",
                    x_order=None, hue_order=None, hue_colors=None,
                    figsize=None):
    """Grouped bar chart (1 x N_metrics) with error bars.

    Parameters
    ----------
    metrics : list of (col_name, display_label)
    """
    if x_order is None:
        x_order = COND_ORDER
    if hue_order is None:
        hue_order = ATTN_ORDER
    if hue_colors is None:
        hue_colors = ATTN_COLORS

    n_metrics = len(metrics)
    if figsize is None:
        figsize = (4.5 * n_metrics, 4.5)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, constrained_layout=True)
    if n_metrics == 1:
        axes = [axes]
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    n_hue = len(hue_order)
    bar_width = 0.8 / n_hue
    x = np.arange(len(x_order))

    for ax, (col, label) in zip(axes, metrics):
        ax_yvals = []
        for hi, hue_val in enumerate(hue_order):
            means, stds = [], []
            for xv in x_order:
                if x_var == "Condition":
                    vals = final_df[(final_df["Condition"] == xv) & (final_df["Attention"] == hue_val)][col].dropna().values
                else:
                    vals = final_df[(final_df["Attention"] == xv) & (final_df["Condition"] == hue_val)][col].dropna().values
                means.append(np.mean(vals) if len(vals) > 0 else 0)
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)

            ax_yvals.extend([m + s for m, s in zip(means, stds)])
            ax_yvals.extend([m - s for m, s in zip(means, stds)])

            offset = (hi - (n_hue - 1) / 2) * bar_width
            ax.bar(x + offset, means, bar_width * 0.88, yerr=stds,
                   label=hue_val, color=hue_colors[hue_val],
                   edgecolor="black", linewidth=0.4, capsize=3,
                   error_kw={"linewidth": 0.7})

        # Zoom y-axis to data range for better visibility
        if ax_yvals:
            ymin, ymax = min(ax_yvals), max(ax_yvals)
            rng = ymax - ymin
            if rng > 0:
                ax.set_ylim(ymin - rng * 0.20, ymax + rng * 0.20)

        ax.set_xticks(x)
        ax.set_xticklabels(x_order, fontsize=9)
        ax.set_title(label)
        ax.set_ylabel(label)
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_hue,
               bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=9)

    _savefig(fig, out_dir, out_stem)
    plt.close(fig)


# ====================================================================
# Figure 1 -- Long Context Extension
# ====================================================================
def plot_figure1_lines(data, out_dir):
    """2x2 grid: valid_long Loss, Accuracy, MCC, F1 over training steps."""
    panels = [
        ("valid_long_loss", "Valid Long Loss", "Loss"),
        ("valid_long_accuracy", "Valid Long Accuracy", "Accuracy"),
        ("valid_long_mcc", "Valid Long MCC", "MCC"),
        ("valid_long_f1", "Valid Long F1", "F1"),
    ]
    _plot_line_grid(data, panels,
                    suptitle="Long Context Extension (1024 tokens)",
                    out_dir=out_dir, out_stem="figure1_long_context_lines")


def plot_figure1_bars(final_df, out_dir):
    """Bar chart: final test metrics grouped by condition, hue = attention."""
    metrics = [
        ("prot_test_loss", "Test Loss"),
        ("prot_test_acc", "Test Accuracy"),
        ("prot_test_mcc", "Test MCC"),
    ]
    _plot_bar_chart(final_df, metrics,
                    suptitle="Final Test Metrics — Extended Context (1024 tokens)",
                    out_dir=out_dir, out_stem="figure1_test_bars",
                    x_var="Condition", hue_var="Attention",
                    x_order=COND_ORDER, hue_order=ATTN_ORDER,
                    hue_colors=ATTN_COLORS, figsize=(14, 4.5))


# ====================================================================
# Figure 2 -- Training Context Learning
# ====================================================================
def plot_figure2_lines(data, out_dir):
    """2x2 grid: Train Loss, Valid Short Loss / Accuracy / MCC."""
    panels = [
        ("train_loss", "Train Loss", "Loss"),
        ("valid_short_loss", "Valid Short Loss", "Loss"),
        ("valid_short_accuracy", "Valid Short Accuracy", "Accuracy"),
        ("valid_short_mcc", "Valid Short MCC", "MCC"),
    ]
    _plot_line_grid(data, panels,
                    suptitle="Training Context Learning (256 tokens)",
                    out_dir=out_dir, out_stem="figure2_training_context_lines")


def plot_figure2_bars(final_df, out_dir):
    """Bar chart: best valid-short metrics grouped by condition, hue = attention."""
    metrics = [
        ("prot_best_valid_short_loss", "Valid Short Loss"),
        ("prot_best_valid_short_acc", "Valid Short Accuracy"),
        ("prot_best_valid_short_mcc", "Valid Short MCC"),
    ]
    _plot_bar_chart(final_df, metrics,
                    suptitle="Best Validation Metrics — Training Context (256 tokens)",
                    out_dir=out_dir, out_stem="figure2_valid_short_bars",
                    x_var="Condition", hue_var="Attention",
                    x_order=COND_ORDER, hue_order=ATTN_ORDER,
                    hue_colors=ATTN_COLORS, figsize=(14, 4.5))


# ====================================================================
# Figure 3 -- DroPE Recovery
# ====================================================================
def plot_figure3_lines(data, out_dir):
    """2x2 grid: valid-short and valid-long Loss / Accuracy for RoPE vs RoPE-off.

    Only the two rotary conditions are plotted; a vertical dashed line marks
    the step at which RoPE is dropped.
    """
    panels = [
        ("valid_short_loss", "Valid Short Loss (256 tok)", "Loss"),
        ("valid_long_loss", "Valid Long Loss (1024 tok)", "Loss"),
        ("valid_short_accuracy", "Valid Short Accuracy", "Accuracy"),
        ("valid_long_accuracy", "Valid Long Accuracy", "Accuracy"),
    ]
    _plot_line_grid(data, panels,
                    suptitle="DroPE Recovery: RoPE vs RoPE-off",
                    out_dir=out_dir, out_stem="figure3_drop_recovery_lines",
                    conditions=["RoPE", "RoPE-off"],
                    show_drop_line=True)


def plot_figure3_bars(final_df, out_dir):
    """Bar chart: RoPE vs RoPE-off final metrics, grouped by attention type.

    Adds significance brackets (Welch t-test) within each attention-type group.
    """
    rope_df = final_df[final_df["Condition"].isin(["RoPE", "RoPE-off"])].copy()

    metrics = [
        ("prot_test_loss", "Test Loss"),
        ("prot_test_acc", "Test Accuracy"),
        ("prot_test_mcc", "Test MCC"),
    ]

    n_met = len(metrics)
    fig, axes = plt.subplots(1, n_met, figsize=(14, 4.5), constrained_layout=True)
    fig.suptitle("DroPE Recovery — Final Test Metrics", fontsize=13, fontweight="bold")

    hue_order = ["RoPE", "RoPE-off"]
    n_hue = len(hue_order)
    bar_width = 0.8 / n_hue
    x = np.arange(len(ATTN_ORDER))

    for ax, (col, label) in zip(axes, metrics):
        ax_yvals = []
        for hi, cond in enumerate(hue_order):
            means, stds = [], []
            for attn in ATTN_ORDER:
                vals = rope_df[(rope_df["Attention"] == attn) & (rope_df["Condition"] == cond)][col].dropna().values
                means.append(np.mean(vals) if len(vals) > 0 else 0)
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
            ax_yvals.extend([m + s for m, s in zip(means, stds)])
            ax_yvals.extend([m - s for m, s in zip(means, stds)])
            offset = (hi - (n_hue - 1) / 2) * bar_width
            ax.bar(x + offset, means, bar_width * 0.88, yerr=stds,
                   label=cond, color=COND_COLORS[cond],
                   edgecolor="black", linewidth=0.4, capsize=3,
                   error_kw={"linewidth": 0.7})

        # Zoom y-axis into data range before adding brackets
        if ax_yvals:
            ymin, ymax = min(ax_yvals), max(ax_yvals)
            rng = ymax - ymin
            if rng > 0:
                ax.set_ylim(ymin - rng * 0.20, ymax + rng * 0.35)

        # Significance brackets: RoPE vs RoPE-off within each attention group
        y_top = ax.get_ylim()[1]
        ylim_range = y_top - ax.get_ylim()[0]
        y_annot = y_top - ylim_range * 0.12

        for xi, attn in enumerate(ATTN_ORDER):
            v_rope = rope_df[(rope_df["Attention"] == attn) & (rope_df["Condition"] == "RoPE")][col].dropna().values
            v_off = rope_df[(rope_df["Attention"] == attn) & (rope_df["Condition"] == "RoPE-off")][col].dropna().values
            _, p = _welch_ttest(v_rope, v_off)
            stars = _sig_stars(p)
            x1 = xi + (0 - (n_hue - 1) / 2) * bar_width
            x2 = xi + (1 - (n_hue - 1) / 2) * bar_width
            _draw_sig_bracket(ax, x1, x2, y_annot, stars)

        # Re-adjust y-lim to fit brackets
        ax.set_ylim(top=y_annot + ylim_range * 0.10)

        ax.set_xticks(x)
        ax.set_xticklabels(ATTN_ORDER, fontsize=9)
        ax.set_title(label)
        ax.set_ylabel(label)
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_hue,
               bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=9)

    _savefig(fig, out_dir, "figure3_drop_bars")
    plt.close(fig)


# ====================================================================
# Statistical summary table
# ====================================================================
def print_stats_table(final_df):
    """Print a formatted table of mean +/- std for every (Condition, Attention)
    combination, plus Welch t-test p-values comparing Dual Triangle against
    Bidirectional and Causal within each PE condition.
    """
    metric_cols = [
        ("prot_test_loss", "Test Loss"),
        ("prot_test_acc", "Test Acc"),
        ("prot_test_mcc", "Test MCC"),
        ("prot_best_valid_short_loss", "VShort Loss"),
        ("prot_best_valid_short_acc", "VShort Acc"),
        ("prot_best_valid_short_mcc", "VShort MCC"),
        ("prot_best_valid_long_loss", "VLong Loss"),
        ("prot_best_valid_long_acc", "VLong Acc"),
        ("prot_best_valid_long_mcc", "VLong MCC"),
    ]

    sep = "-" * 130
    print("\n" + sep)
    print("STATISTICAL SUMMARY  (mean +/- std across seeds 11, 22, 33)")
    print(sep)

    for col, col_label in metric_cols:
        print(f"\n{'=' * 130}")
        print(f"  {col_label}  ({col})")
        print(f"{'=' * 130}")
        header = f"  {'Condition':<12} | {'Dual Triangle':>20} | {'Bidirectional':>20} | {'Causal':>20} | {'DT v Bidir':>14} | {'DT v Causal':>14}"
        print(header)
        print(f"  {'-' * 12}-+-{'-' * 20}-+-{'-' * 20}-+-{'-' * 20}-+-{'-' * 14}-+-{'-' * 14}")

        for cond in COND_ORDER:
            vals = {}
            for attn in ATTN_ORDER:
                v = final_df[(final_df["Attention"] == attn) & (final_df["Condition"] == cond)][col].dropna().values
                vals[attn] = v

            strs = {}
            for attn in ATTN_ORDER:
                v = vals[attn]
                if len(v) > 0:
                    strs[attn] = f"{np.mean(v):.4f} +/- {np.std(v, ddof=1):.4f}"
                else:
                    strs[attn] = "N/A"

            _, p_bidir = _welch_ttest(vals["Dual Triangle"], vals["Bidirectional"])
            _, p_causal = _welch_ttest(vals["Dual Triangle"], vals["Causal"])

            p_bidir_str = f"p={p_bidir:.3f} {_sig_stars(p_bidir)}"
            p_causal_str = f"p={p_causal:.3f} {_sig_stars(p_causal)}"

            print(f"  {cond:<12} | {strs['Dual Triangle']:>20} | {strs['Bidirectional']:>20} | {strs['Causal']:>20} | {p_bidir_str:>14} | {p_causal_str:>14}")

    # DroPE comparison: RoPE vs RoPE-off within each attention type
    print(f"\n{'=' * 130}")
    print("  DroPE Recovery:  RoPE vs RoPE-off (Welch t-test)")
    print(f"{'=' * 130}")
    header2 = f"  {'Metric':<16} | {'Dual Triangle':>20} | {'Bidirectional':>20} | {'Causal':>20}"
    print(header2)
    print(f"  {'-' * 16}-+-{'-' * 20}-+-{'-' * 20}-+-{'-' * 20}")

    for col, col_label in metric_cols:
        parts = []
        for attn in ATTN_ORDER:
            v_rope = final_df[(final_df["Attention"] == attn) & (final_df["Condition"] == "RoPE")][col].dropna().values
            v_off = final_df[(final_df["Attention"] == attn) & (final_df["Condition"] == "RoPE-off")][col].dropna().values
            _, p = _welch_ttest(v_rope, v_off)
            parts.append(f"p={p:.3f} {_sig_stars(p)}")
        print(f"  {col_label:<16} | {parts[0]:>20} | {parts[1]:>20} | {parts[2]:>20}")

    print(sep + "\n")


# ====================================================================
# Main
# ====================================================================
def main():
    _style()

    root = Path(__file__).resolve().parent.parent.parent  # repo root
    data_dir = root / "results" / "all"
    out_dir = root / "results" / "combined_plots"

    results_dirs = [
        root / "results" / "group_protein_non",
        root / "results" / "rotary",
        root / "results" / "rotary_drop",
    ]

    print("Loading wandb CSVs ...")
    data = load_wandb_csvs(data_dir)

    print("Loading final results ...")
    final_df = load_final_results(results_dirs)
    print(f"  {len(final_df)} rows loaded from {len(results_dirs)} result files.\n")

    print("Generating Figure 1 -- Long Context Extension ...")
    plot_figure1_lines(data, out_dir)
    plot_figure1_bars(final_df, out_dir)

    print("Generating Figure 2 -- Training Context Learning ...")
    plot_figure2_lines(data, out_dir)
    plot_figure2_bars(final_df, out_dir)

    print("Generating Figure 3 -- DroPE Recovery ...")
    plot_figure3_lines(data, out_dir)
    plot_figure3_bars(final_df, out_dir)

    print_stats_table(final_df)

    print("Done. All figures saved to:", out_dir)


if __name__ == "__main__":
    main()
