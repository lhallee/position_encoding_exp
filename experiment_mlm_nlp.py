"""Experiment 2a: Masked Language Modeling on natural language (FineWeb-Edu).

Trains UNet transformer models from scratch on natural language data,
comparing attention types and positional encoding strategies.
"""

import src.entrypoint_setup

import argparse
import subprocess
import sys
import pandas as pd
from pathlib import Path

from src.plotting.plot_mlm import plot_all


COMMAND_ARGS = [
    "--dataset nl --conditions none --out_dir group_nl_1",
    "--dataset nl --conditions rotary --out_dir group_nl_2",
    "--dataset nl --conditions rotary_drop --out_dir group_nl_3",
]


def _python_cmd() -> str:
    return "py" if sys.platform.startswith("win") else "python"


def _build_commands(wandb_token: str | None = None, seeds: list[int] | None = None) -> list[str]:
    py_cmd = _python_cmd()
    extra_args = ""
    if wandb_token is not None:
        extra_args += f" --wandb_token {wandb_token} --wandb_project pos-encoding-mlm-nlp"
    if seeds is not None:
        extra_args += " --seeds " + " ".join(str(s) for s in seeds)
    return [f"{py_cmd} -m src.training.sweep_mlm {args}{extra_args}" for args in COMMAND_ARGS]


def run_experiments(wandb_token: str | None = None, seeds: list[int] | None = None) -> None:
    commands = _build_commands(wandb_token=wandb_token, seeds=seeds)
    for i, cmd in enumerate(commands, 1):
        print(f"\n--- Running experiment {i}/{len(commands)} ---")
        print(f"Command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


def compile_and_plot(out_dir_path: Path) -> None:
    root = Path(".")
    group_dirs = sorted(list(root.glob("group_nl_*")))

    if not group_dirs:
        print("No group_nl_* directories found in current path.")
        return

    print(f"Found {len(group_dirs)} result directories.")

    result_dfs = []
    history_dfs = []
    for d in group_dirs:
        if not d.is_dir():
            continue
        results_file = d / "results.csv"
        history_file = d / "history.csv"
        if results_file.exists():
            print(f"Reading {results_file}")
            result_dfs.append(pd.read_csv(results_file))
        if history_file.exists():
            print(f"Reading {history_file}")
            history_dfs.append(pd.read_csv(history_file))

    if not result_dfs or not history_dfs:
        print("No usable results.csv or history.csv files were found.")
        return

    results_df = pd.concat(result_dfs, ignore_index=True)
    history_df = pd.concat(history_dfs, ignore_index=True)

    config_cols = [
        "attention_type", "positional_mode", "drop_positions_tokens",
        "hidden_size", "n_layers", "head_size", "train_seq_len", "test_seq_len", "seed",
    ]
    present_cols = [c for c in config_cols if c in results_df.columns]
    if present_cols:
        before = len(results_df)
        results_df = results_df.drop_duplicates(subset=present_cols)
        after = len(results_df)
        if before > after:
            print(f"Removed {before - after} duplicate result rows.")

    out_dir_path.mkdir(parents=True, exist_ok=True)
    combined_results_csv = out_dir_path / "results.csv"
    combined_history_csv = out_dir_path / "history.csv"
    results_df.to_csv(combined_results_csv, index=False)
    history_df.to_csv(combined_history_csv, index=False)
    print(f"Saved {len(results_df)} combined result rows to {combined_results_csv}")

    plots_dir = out_dir_path / "plots"
    print(f"Generating plots in {plots_dir}...")
    plot_all(results_csv=combined_results_csv, history_csv=combined_history_csv, out_dir=plots_dir)
    print("Plotting complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 2a (MLM on NLP data) and compile plots.")
    parser.add_argument("--out_dir", type=str, default="outputs_exp2_nl", help="Output directory.")
    parser.add_argument("--skip_runs", action="store_true", help="Skip running experiments and only compile/plot.")
    parser.add_argument("--bugfix", action="store_true", help="Run a cheap bugfix sweep with checks.")
    parser.add_argument("--wandb_token", type=str, default=None, help="Weights & Biases API token. Enables wandb logging if provided.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11], help="Random seeds to pass to sweep runs.")
    args = parser.parse_args()

    if args.bugfix:
        from src.training.bugfix_checks import run_experiment2_checks

        run_experiment2_checks()
        py_cmd = _python_cmd()
        wandb_args = ""
        if args.wandb_token is not None:
            wandb_args = f" --wandb_token {args.wandb_token} --wandb_project pos-encoding-mlm-nlp"
        cmd = (
            f"{py_cmd} -m src.training.sweep_mlm "
            "--dataset nl "
            "--out_dir outputs_bugfix_exp2_nl "
            "--total_tokens 128 --warmup_tokens 32 --cooldown_tokens 32 --eval_every 2 "
            "--batch_size 2 "
            "--train_seq_len 16 --test_seq_len 32 "
            "--hidden_size 32 --n_layers 2 "
            "--seeds 11 "
            "--conditions none "
            "--valid_size 4 --test_size 4 --shuffle_buffer 100 "
            "--no_unet --no_bfloat16 "
            f"{wandb_args}"
        )
        print(f"Bugfix command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    elif not args.skip_runs:
        run_experiments(wandb_token=args.wandb_token, seeds=args.seeds)
    else:
        print("Skipping experiment execution as requested.")

    if args.bugfix:
        out_dir = Path("outputs_bugfix_exp2_nl")
        results_csv = out_dir / "results.csv"
        history_csv = out_dir / "history.csv"
        if results_csv.exists() and history_csv.exists():
            plot_all(results_csv=results_csv, history_csv=history_csv, out_dir=out_dir / "plots")
    else:
        compile_and_plot(Path(args.out_dir))
    print("\n--- All tasks completed ---")


if __name__ == "__main__":
    main()
