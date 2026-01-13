from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.experiment.plot_results import plot_all
from src.experiment.train_one import TrainConfig, train_one
from src.models.transformer import TransformerConfig
from src.utils.seed import set_global_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run position-probe sweeps (causal vs bidirectional, pos modes).")
    p.add_argument("--out_dir", type=str, default="outputs", help="Output directory.")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--steps", type=int, default=1500, help="Training steps per run.")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    p.add_argument("--eval_batches", type=int, default=50, help="Evaluation batches.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds.")
    p.add_argument("--seq_len", type=int, default=128, help="Sequence length.")
    p.add_argument("--vocab_size", type=int, default=128, help="Vocab size (token IDs 1..128).")
    p.add_argument("--progress", action="store_true", help="Show per-run training progress bars.")
    return p.parse_args()


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"device must be auto|cpu|cuda, got {device_arg}")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = _device_from_arg(args.device)
    set_global_seed(0)

    # Fixed modeling choices for simplicity:
    # - n_heads scales with d_model (keep head dim ~32 where possible)
    # - d_ff = 4 * d_model
    d_models = [32, 64, 128, 256]
    n_layers_list = [1, 2, 4, 6]

    # Conditions:
    # - positional_mode=none (no PE ever)
    # - positional_mode=learned_abs (PE always)
    # - positional_mode=learned_abs + drop_positions_step (DroPE-like)
    conditions: list[tuple[str, int]] = [
        ("none", -1),
        ("learned_abs", -1),
        ("learned_abs", int(args.steps * 2 / 3)),
    ]
    attention_types = ["bidirectional", "causal"]

    rows: list[dict[str, float | int | str]] = []
    total_runs = len(args.seeds) * len(attention_types) * len(conditions) * len(d_models) * len(n_layers_list)
    run_idx = 0

    for seed in args.seeds:
        for attention_type in attention_types:
            for positional_mode, drop_step in conditions:
                for d_model in d_models:
                    for n_layers in n_layers_list:
                        run_idx += 1

                        # keep head dim reasonable; ensure divisible
                        if d_model >= 128:
                            n_heads = 4
                        else:
                            n_heads = 2
                        if d_model % n_heads != 0:
                            continue

                        model_cfg = TransformerConfig(
                            vocab_size=args.vocab_size,
                            seq_len=args.seq_len,
                            d_model=d_model,
                            n_layers=n_layers,
                            n_heads=n_heads,
                            d_ff=4 * d_model,
                            dropout=0.0,
                            attention_type=attention_type,
                            positional_mode=positional_mode,
                        )
                        train_cfg = TrainConfig(
                            steps=args.steps,
                            batch_size=args.batch_size,
                            lr=3e-4,
                            weight_decay=0.01,
                            eval_batches=args.eval_batches,
                            drop_positions_step=None if drop_step < 0 else drop_step,
                        )

                        print(
                            f"[{run_idx}/{total_runs}] seed={seed} attn={attention_type} pos={positional_mode} "
                            f"drop={drop_step} layers={n_layers} d={d_model} device={device}"
                        )
                        row = train_one(
                            model_cfg=model_cfg,
                            train_cfg=train_cfg,
                            seed=seed,
                            device=device,
                            vocab_low_inclusive=1,
                            vocab_high_inclusive=128,
                            progress=args.progress,
                        )
                        rows.append(row)

                        df = pd.DataFrame(rows)
                        df.to_csv(out_dir / "results.csv", index=False)

    plot_all(results_csv=out_dir / "results.csv", out_dir=plots_dir)
    print(f"Done. Wrote {out_dir / 'results.csv'} and plots to {plots_dir}")


if __name__ == "__main__":
    main()

