import src.entrypoint_setup

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.experiment.plot_results import plot_all
from src.experiment.train_one import TrainConfig, train_one
from src.models.transformer import TransformerConfig
from src.utils.seed import set_global_seed



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run position-probe sweeps (causal vs bidirectional, pos modes).")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--steps", type=int, default=512, help="Minibatches per evaluation (steps-per-eval).")
    parser.add_argument("--max_evals", type=int, default=25, help="Max #eval cycles before stopping.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on eval accuracy.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--eval_batches", type=int, default=32, help="Evaluation batches.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds.")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length.")
    parser.add_argument("--vocab_size", type=int, default=64, help="Vocab size (token IDs 1..vocab_size).")
    parser.add_argument("--d_models", type=int, nargs="+", default=[32, 64, 128, 256], help="List of d_model values.")
    parser.add_argument("--n_layers", type=int, nargs="+", default=[1, 2, 4, 6], help="List of n_layers values.")
    parser.add_argument("--progress", action="store_true", help="Show per-run training progress bars.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability (default: 0.0).")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA (speed).")
    parser.add_argument("--flush_every", type=int, default=10, help="Write results.csv every N runs (0=only at end).")
    return parser.parse_args()


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # - d_ff = 3 * d_model
    d_models = sorted(list(args.d_models), reverse=True)
    n_layers_list = sorted(list(args.n_layers), reverse=True)

    # Conditions:
    # - positional_mode=none (no PE ever)
    # - positional_mode=learned_abs (PE always)
    # - positional_mode=learned_abs + drop_positions_step (DroPE-like)
    # - positional_mode=rotary (RoPE applied inside attention)
    # - positional_mode=rotary + drop_positions_step (drop RoPE after eval #2)
    conditions: list[tuple[str, int]] = [
        ("none", -1),
        ("learned_abs", -1),
        # Drop after the 2nd evaluation (i.e., after 2 * steps_per_eval minibatches).
        ("learned_abs", int(2 * args.steps)),
        ("rotary", -1),
        ("rotary", int(2 * args.steps)),
    ]
    attention_types = ["bidirectional", "causal", "dual_triangle"]

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
                        elif d_model >= 64:
                            n_heads = 2
                        else:
                            n_heads = 1
                        if d_model % n_heads != 0:
                            continue

                        model_cfg = TransformerConfig(
                            vocab_size=args.vocab_size,
                            seq_len=args.seq_len,
                            d_model=d_model,
                            n_layers=n_layers,
                            n_heads=n_heads,
                            d_ff=3 * d_model,
                            dropout=float(args.dropout),
                            attention_type=attention_type,
                            positional_mode=positional_mode,
                        )
                        train_cfg = TrainConfig(
                            steps_per_eval=args.steps,
                            max_evals=args.max_evals,
                            patience=args.patience,
                            warmup_steps=args.steps,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            eval_batches=args.eval_batches,
                            drop_positions_step=None if drop_step < 0 else drop_step,
                            label_mode="true",
                            amp=bool(args.amp),
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
                            vocab_high_inclusive=args.vocab_size,
                            progress=args.progress,
                        )
                        print(f"  eval_acc={row['eval_acc']:.4f}")
                        rows.append(row)

                        if int(args.flush_every) > 0 and (len(rows) % int(args.flush_every) == 0):
                            df = pd.DataFrame(rows)
                            df.to_csv(out_dir / "results.csv", index=False)

    # Controls: random labels (i.e., "shuffled dataset labels") on the largest model config.
    # We run len(seeds) controls for each attention type, last.
    control_d_model = max(d_models)
    control_n_layers = max(n_layers_list)
    control_n_heads = 4 if control_d_model >= 128 else 2
    for seed in args.seeds:
        for attention_type in attention_types:
            model_cfg = TransformerConfig(
                vocab_size=args.vocab_size,
                seq_len=args.seq_len,
                d_model=control_d_model,
                n_layers=control_n_layers,
                n_heads=control_n_heads,
                d_ff=4 * control_d_model,
                dropout=float(args.dropout),
                attention_type=attention_type,
                positional_mode="learned_abs",
            )
            train_cfg = TrainConfig(
                steps_per_eval=args.steps,
                max_evals=args.max_evals,
                patience=args.patience,
                warmup_steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                eval_batches=args.eval_batches,
                drop_positions_step=None,
                label_mode="random",
                amp=bool(args.amp),
            )
            print(
                f"[control] seed={seed} attn={attention_type} pos=learned_abs drop=-1 "
                f"layers={control_n_layers} d={control_d_model} device={device} label_mode=random"
            )
            row = train_one(
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                seed=seed,
                device=device,
                vocab_low_inclusive=1,
                vocab_high_inclusive=args.vocab_size,
                progress=args.progress,
            )
            print(f"  eval_acc={row['eval_acc']:.4f}")
            rows.append(row)
            if int(args.flush_every) > 0 and (len(rows) % int(args.flush_every) == 0):
                df = pd.DataFrame(rows)
                df.to_csv(out_dir / "results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)

    plot_all(results_csv=out_dir / "results.csv", out_dir=plots_dir)
    print(f"Done. Wrote {out_dir / 'results.csv'} and plots to {plots_dir}")


if __name__ == "__main__":
    main()

