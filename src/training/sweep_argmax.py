"""Sweep runner for Experiment 1: argmax position probe."""

import src.entrypoint_setup

import argparse
import os
import pandas as pd
import torch
from pathlib import Path

from src.plotting.plot_argmax import plot_all
from src.training.train_argmax import TrainConfig, train_one
from src.models.transformer import TransformerConfig
from src.utils.seed import set_global_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run position-probe sweeps (causal vs bidirectional, pos modes).")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--steps", type=int, default=256, help="Minibatches per evaluation (steps-per-eval).")
    parser.add_argument("--max_evals", type=int, default=10, help="Max #eval cycles before stopping.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on eval accuracy.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size.")
    parser.add_argument("--eval_batches", type=int, default=16, help="Evaluation batches.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33], help="Random seeds.")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length.")
    parser.add_argument("--vocab_size", type=int, default=64, help="Vocab size (token IDs 1..vocab_size).")
    parser.add_argument("--d_models", type=int, nargs="+", default=[32, 64, 128, 256], help="List of hidden_size values.")
    parser.add_argument("--n_layers", type=int, nargs="+", default=[1, 2, 4, 6], help="List of n_layers values.")
    parser.add_argument("--progress", action="store_true", help="Show per-run training progress bars.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability (default: 0.0).")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA (speed).")
    parser.add_argument("--flush_every", type=int, default=1, help="Write results.csv every N runs (0=only at end).")
    parser.add_argument(
        "--conditions", type=str, nargs="+",
        default=["none", "learned_abs", "learned_abs_drop", "rotary", "rotary_drop"],
        choices=["none", "learned_abs", "learned_abs_drop", "rotary", "rotary_drop"],
        help="Which positional/Drop conditions to run.",
    )
    parser.add_argument("--wandb_token", type=str, default=None, help="Weights & Biases API token. Enables wandb logging if provided and wandb is available.")
    parser.add_argument("--wandb_project", type=str, default="pos-encoding-argmax", help="Weights & Biases project name.")
    return parser.parse_args()


def _init_wandb(args: argparse.Namespace) -> bool:
    """Login to wandb if token is provided and wandb is available. Returns True if wandb is active."""
    if args.wandb_token is None:
        return False
    assert os.environ["WANDB_AVAILABLE"] == "true", (
        "--wandb_token was provided but wandb is not installed. Install with: pip install wandb"
    )
    import wandb
    wandb.login(key=args.wandb_token)
    return True


def _start_wandb_run(*, project: str, config: dict, run_name: str):
    """Create and return a new wandb run."""
    import wandb
    return wandb.init(project=project, config=config, name=run_name, reinit=True)


def _finish_wandb_run(wandb_run, summary: dict) -> None:
    """Log final summary metrics and finish the wandb run."""
    for k, v in summary.items():
        wandb_run.summary[k] = v
    wandb_run.finish()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = _init_wandb(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(0)

    d_models = sorted(list(args.d_models), reverse=True)
    n_layers_list = sorted(list(args.n_layers), reverse=True)

    drop_after = int(2 * args.steps)
    condition_map: dict[str, tuple[str, int]] = {
        "none": ("none", -1),
        "learned_abs": ("learned_abs", -1),
        "learned_abs_drop": ("learned_abs", drop_after),
        "rotary": ("rotary", -1),
        "rotary_drop": ("rotary", drop_after),
    }
    conditions: list[tuple[str, int]] = [condition_map[name] for name in args.conditions]
    attention_types = ["bidirectional", "causal", "dual_triangle"]

    rows: list[dict[str, float | int | str]] = []
    total_runs = len(args.seeds) * len(attention_types) * len(conditions) * len(d_models) * len(n_layers_list)
    run_idx = 0

    for seed in args.seeds:
        for attention_type in attention_types:
            for positional_mode, drop_step in conditions:
                for hidden_size in d_models:
                    for n_layers in n_layers_list:
                        run_idx += 1

                        head_size = 128 if attention_type == "dual_triangle" else 64
                        if head_size > hidden_size:
                            head_size = hidden_size

                        model_cfg = TransformerConfig(
                            vocab_size=args.vocab_size,
                            seq_len=args.seq_len,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            head_size=head_size,
                            intermediate_size=3 * hidden_size,
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
                            f"drop={drop_step} layers={n_layers} d={hidden_size} device={device}"
                        )

                        wandb_run = None
                        if use_wandb:
                            run_name = f"s{seed}_{attention_type}_{positional_mode}_drop{drop_step}_L{n_layers}_d{hidden_size}"
                            wandb_run = _start_wandb_run(
                                project=args.wandb_project,
                                config={
                                    "seed": seed, "attention_type": attention_type,
                                    "positional_mode": positional_mode, "drop_positions_step": drop_step,
                                    "n_layers": n_layers, "hidden_size": hidden_size,
                                    "head_size": head_size, "seq_len": args.seq_len,
                                    "vocab_size": args.vocab_size, "label_mode": "true",
                                    "steps_per_eval": args.steps, "max_evals": args.max_evals,
                                    "patience": args.patience, "batch_size": args.batch_size,
                                    "lr": args.lr, "weight_decay": args.weight_decay,
                                },
                                run_name=run_name,
                            )

                        row = train_one(
                            model_cfg=model_cfg,
                            train_cfg=train_cfg,
                            seed=seed,
                            device=device,
                            vocab_low_inclusive=1,
                            vocab_high_inclusive=args.vocab_size,
                            progress=args.progress,
                            wandb_run=wandb_run,
                        )

                        if wandb_run is not None:
                            _finish_wandb_run(wandb_run, row)

                        print(f"  eval_acc={row['eval_acc']:.4f}")
                        rows.append(row)

                        if int(args.flush_every) > 0 and (len(rows) % int(args.flush_every) == 0):
                            df = pd.DataFrame(rows)
                            df.to_csv(out_dir / "results.csv", index=False)

    # Controls: random labels
    control_d_model = max(d_models)
    control_n_layers = max(n_layers_list)
    for seed in args.seeds:
        for attention_type in attention_types:
            head_size = 128 if attention_type == "dual_triangle" else 64
            if head_size > control_d_model:
                head_size = control_d_model

            model_cfg = TransformerConfig(
                vocab_size=args.vocab_size,
                seq_len=args.seq_len,
                hidden_size=control_d_model,
                n_layers=control_n_layers,
                head_size=head_size,
                intermediate_size=4 * control_d_model,
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

            wandb_run = None
            if use_wandb:
                run_name = f"control_s{seed}_{attention_type}_random_L{control_n_layers}_d{control_d_model}"
                wandb_run = _start_wandb_run(
                    project=args.wandb_project,
                    config={
                        "seed": seed, "attention_type": attention_type,
                        "positional_mode": "learned_abs", "drop_positions_step": -1,
                        "n_layers": control_n_layers, "hidden_size": control_d_model,
                        "head_size": head_size, "seq_len": args.seq_len,
                        "vocab_size": args.vocab_size, "label_mode": "random",
                        "steps_per_eval": args.steps, "max_evals": args.max_evals,
                        "patience": args.patience, "batch_size": args.batch_size,
                        "lr": args.lr, "weight_decay": args.weight_decay,
                    },
                    run_name=run_name,
                )

            row = train_one(
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                seed=seed,
                device=device,
                vocab_low_inclusive=1,
                vocab_high_inclusive=args.vocab_size,
                progress=args.progress,
                wandb_run=wandb_run,
            )

            if wandb_run is not None:
                _finish_wandb_run(wandb_run, row)

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
