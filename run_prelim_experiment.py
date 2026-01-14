import src.entrypoint_setup

import pandas as pd
import subprocess
import argparse
from pathlib import Path

from src.experiment.plot_results import plot_all


# List of commands extracted from src/experiment/run_sweep.py
COMMANDS = [
    "python -m src.experiment.run_sweep --progress --d_models 4 16 64 256 --n_layers 1 2 4 --conditions none --out_dir group_1",
    "python -m src.experiment.run_sweep --progress --d_models 512 1024 --n_layers 1 2 4 --conditions none --out_dir group_2",
    "python -m src.experiment.run_sweep --progress --d_models 4 16 64 256 512 1024 --n_layers 8 12 --conditions none --out_dir group_3",
    "python -m src.experiment.run_sweep --progress --d_models 4 16 64 256 --n_layers 1 2 4 --conditions learned_abs --out_dir group_4",
    "python -m src.experiment.run_sweep --progress --d_models 512 1024 --n_layers 1 2 4 --conditions learned_abs --out_dir group_5",
    "python -m src.experiment.run_sweep --progress --d_models 4 16 64 256 512 1024 --n_layers 8 12 --conditions learned_abs --out_dir group_6",
    "python -m src.experiment.run_sweep --progress --d_models 4 16 64 256 --n_layers 1 2 4 --conditions learned_abs_drop --out_dir group_7",
    "python -m src.experiment.run_sweep --progress --d_models 512 1024 --n_layers 1 2 4 --conditions learned_abs_drop --out_dir group_8",
    "python -m src.experiment.run_sweep --progress --d_models 4 16 64 256 512 1024 --n_layers 8 12 --conditions learned_abs_drop --out_dir group_9",
    "python -m src.experiment.run_sweep --progress --d_models 4 16 64 256 --n_layers 1 2 4 --conditions rotary --out_dir group_10",
    "python -m src.experiment.run_sweep --progress --d_models 512 1024 --n_layers 1 2 4 --conditions rotary --out_dir group_11",
    "python -m src.experiment.run_sweep --progress --d_models 4 16 64 256 512 1024 --n_layers 8 12 --conditions rotary --out_dir group_12",
    "python -m src.experiment.run_sweep --progress --d_models 4 16 64 256 --n_layers 1 2 4 --conditions rotary_drop --out_dir group_13",
    "python -m src.experiment.run_sweep --progress --d_models 512 1024 --n_layers 1 2 4 --conditions rotary_drop --out_dir group_14",
    "python -m src.experiment.run_sweep --progress --d_models 4 16 64 256 512 1024 --n_layers 8 12 --conditions rotary_drop --out_dir group_15",
]


def run_experiments():
    for i, cmd in enumerate(COMMANDS, 1):
        print(f"\n--- Running experiment {i}/{len(COMMANDS)} ---")
        print(f"Command: {cmd}")
        # Note: Using shell=True for convenience with the 'py -m ...' format on Windows
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
            try:
                df = pd.read_csv(results_file)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {results_file}: {e}")

    if not all_dfs:
        print("No results.csv files were found or could be read.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    config_cols = [
        "attention_type", "positional_mode", "drop_positions_step", 
        "label_mode", "n_layers", "d_model", "seed"
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
    try:
        plot_all(results_csv=combined_csv, out_dir=plots_dir)
        print("Plotting complete.")
    except Exception as e:
        print(f"Error during plotting: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run all experiments and compile results.")
    parser.add_argument("--out_dir", type=str, default="compiled_results", help="Directory to save combined results and plots.")
    parser.add_argument("--skip_runs", action="store_true", help="Skip running experiments and only compile/plot.")
    args = parser.parse_args()

    if not args.skip_runs:
        run_experiments()
    else:
        print("Skipping experiment execution as requested.")

    compile_and_plot(Path(args.out_dir))
    print("\n--- All tasks completed ---")

if __name__ == "__main__":
    main()
