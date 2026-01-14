import src.entrypoint_setup

import argparse
import subprocess
import sys
from pathlib import Path

from src.experiment.plot_results_mlm import plot_all


def run_experiment() -> None:
    py_cmd = "py" if sys.platform.startswith("win") else "python"
    cmd = (
        f"{py_cmd} -m src.experiment.run_sweep_mlm --progress "
        "--out_dir outputs_exp2"
    )
    print(f"Command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def compile_and_plot(out_dir_path: Path) -> None:
    results_csv = out_dir_path / "results.csv"
    history_csv = out_dir_path / "history.csv"
    if not results_csv.exists() or not history_csv.exists():
        print("Missing results.csv or history.csv; run experiment2 first.")
        return
    plots_dir = out_dir_path / "plots"
    plot_all(results_csv=results_csv, history_csv=history_csv, out_dir=plots_dir)
    print(f"Saved plots to {plots_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 2 (MLM) and compile plots.")
    parser.add_argument("--out_dir", type=str, default="outputs_exp2", help="Output directory.")
    parser.add_argument("--skip_runs", action="store_true", help="Skip running experiments and only plot.")
    parser.add_argument("--bugfix", action="store_true", help="Run a cheap bugfix sweep with checks.")
    args = parser.parse_args()

    if args.bugfix:
        from src.experiment.bugfix_checks import run_experiment2_checks

        run_experiment2_checks()
        py_cmd = "py" if sys.platform.startswith("win") else "python"
        cmd = (
            f"{py_cmd} -m src.experiment.run_sweep_mlm --progress "
            "--out_dir outputs_bugfix_exp2 "
            "--device cpu "
            "--steps 2 --max_evals_nl 1 --max_evals_prot 1 --patience 0 "
            "--batch_size 2 --eval_batches 1 "
            "--train_seq_len 16 --test_seq_len 32 "
            "--d_model 32 --n_layers 1 "
            "--seeds 11 "
            "--conditions none "
            "--fineweb_valid_docs 4 --fineweb_test_docs 4 --shuffle_buffer 100 "
        )
        print(f"Bugfix command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    elif not args.skip_runs:
        run_experiment()
    else:
        print("Skipping experiment execution as requested.")

    out_dir = Path("outputs_bugfix_exp2") if args.bugfix else Path(args.out_dir)
    compile_and_plot(out_dir)
    print("\n--- All tasks completed ---")


if __name__ == "__main__":
    main()
