"""Experiment 1: Argmax position probe.

Tests whether different attention types can learn positional information
by predicting which position contains the maximum token ID.
"""

import src.entrypoint_setup

import argparse
import subprocess
import sys
import pandas as pd
from pathlib import Path

from src.plotting.plot_argmax import plot_all


COMMAND_ARGS = [
    "--d_models 4 64 768 --n_layers 1 4 12 --conditions none --out_dir group_1",
    #"--d_models 4 64 768 --n_layers 1 4 12 --conditions learned_abs --out_dir group_2",
    #"--d_models 4 64 768 --n_layers 1 4 12 --conditions learned_abs_drop --out_dir group_3",
    #"--d_models 4 64 768 --n_layers 1 4 12 --conditions rotary --out_dir group_4",
    #"--d_models 4 64 768 --n_layers 1 4 12 --conditions rotary_drop --out_dir group_5",
]


def _python_cmd() -> str:
    return "py" if sys.platform.startswith("win") else "python"


def _build_commands(wandb_token: str | None = None) -> list[str]:
    py_cmd = _python_cmd()
    wandb_args = ""
    if wandb_token is not None:
        wandb_args = f" --wandb_token {wandb_token} --wandb_project pos-encoding-argmax"
    return [f"{py_cmd} -m src.training.sweep_argmax {args}{wandb_args}" for args in COMMAND_ARGS]


def run_experiments(wandb_token: str | None = None):
    commands = _build_commands(wandb_token=wandb_token)
    for i, cmd in enumerate(commands, 1):
        print(f"\n--- Running experiment {i}/{len(commands)} ---")
        print(f"Command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


def compile_and_plot(out_dir_path: Path):
    root = Path(".")
    group_dirs = sorted(list(root.glob("group_*")))

    if not group_dirs:
        print("No group_* directories found in current path.")
        return

    print(f"Found {len(group_dirs)} result directories.")

    all_dfs = []
    for d in group_dirs:
        if not d.is_dir():
            continue
        results_file = d / "results.csv"
        if results_file.exists():
            print(f"Reading {results_file}")
            all_dfs.append(pd.read_csv(results_file))

    if not all_dfs:
        print("No results.csv files were found or could be read.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    config_cols = [
        "attention_type", "positional_mode", "drop_positions_step",
        "label_mode", "n_layers", "hidden_size", "head_size", "seed"
    ]
    present_cols = [c for c in config_cols if c in combined_df.columns]

    if present_cols:
        before = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=present_cols)
        after = len(combined_df)
        if before > after:
            print(f"Removed {before - after} duplicate result rows.")

    out_dir_path.mkdir(parents=True, exist_ok=True)
    combined_csv = out_dir_path / "results.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"Saved {len(combined_df)} combined result rows to {combined_csv}")

    plots_dir = out_dir_path / "plots"
    print(f"Generating plots in {plots_dir}...")
    plot_all(results_csv=combined_csv, out_dir=plots_dir)
    print("Plotting complete.")


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1 (argmax position probe) and compile results.")
    parser.add_argument("--out_dir", type=str, default="compiled_results", help="Directory to save combined results and plots.")
    parser.add_argument("--skip_runs", action="store_true", help="Skip running experiments and only compile/plot.")
    parser.add_argument("--bugfix", action="store_true", help="Run a cheap bugfix sweep with checks.")
    parser.add_argument("--wandb_token", type=str, default=None, help="Weights & Biases API token. Enables wandb logging if provided.")
    args = parser.parse_args()

    if args.bugfix:
        from src.training.bugfix_checks import run_experiment1_checks

        run_experiment1_checks()
        py_cmd = _python_cmd()
        wandb_args = ""
        if args.wandb_token is not None:
            wandb_args = f" --wandb_token {args.wandb_token} --wandb_project pos-encoding-argmax"
        cmd = (
            f"{py_cmd} -m src.training.sweep_argmax "
            "--out_dir outputs_bugfix_exp1 "
            "--steps 2 --max_evals 1 --patience 0 "
            "--batch_size 8 --eval_batches 1 "
            "--seeds 11 "
            "--seq_len 16 --vocab_size 32 "
            "--d_models 16 --n_layers 1 "
            "--conditions none "
            f"{wandb_args}"
        )
        print(f"Bugfix command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    elif not args.skip_runs:
        run_experiments(wandb_token=args.wandb_token)
    else:
        print("Skipping experiment execution as requested.")

    out_dir = Path("outputs_bugfix_exp1") if args.bugfix else Path(args.out_dir)
    if args.bugfix:
        results_csv = out_dir / "results.csv"
        plots_dir = out_dir / "plots"
        if results_csv.exists():
            plot_all(results_csv=results_csv, out_dir=plots_dir)
    else:
        compile_and_plot(out_dir)
    print("\n--- All tasks completed ---")


if __name__ == "__main__":
    main()
