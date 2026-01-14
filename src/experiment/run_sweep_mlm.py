import src.entrypoint_setup

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.experiment.plot_results_mlm import plot_all
from src.experiment.train_mlm import MLMTrainConfig, eval_mlm, init_mlm_model, train_mlm_phase
from src.models.transformer import TransformerConfig
from src.tasks.mlm_data import StreamConfig, build_mlm_dataloader
from src.utils.seed import set_global_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLM sweeps for Experiment 2 (NL + proteins).")
    parser.add_argument("--dataset", type=str, default="nl", choices=["nl", "protein"], help="Which dataset to run.")
    parser.add_argument("--out_dir", type=str, default="outputs_exp2", help="Output directory.")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--steps", type=int, default=1000, help="Minibatches per evaluation.")
    parser.add_argument("--max_evals_nl", type=int, default=10, help="Eval cycles on natural language phase.")
    parser.add_argument("--max_evals_prot", type=int, default=10, help="Eval cycles on protein phase.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on eval accuracy.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--eval_batches", type=int, default=8, help="Evaluation batches.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33], help="Random seeds.")
    parser.add_argument("--train_seq_len", type=int, default=128, help="Training sequence length.")
    parser.add_argument("--test_seq_len", type=int, default=512, help="Extended test sequence length.")
    parser.add_argument("--d_model", type=int, default=768, help="Hidden size.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers.")
    parser.add_argument("--progress", action="store_true", help="Show per-run training progress bars.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA (speed).")
    parser.add_argument("--mlm_prob", type=float, default=0.15, help="Masked LM probability.")
    parser.add_argument("--flush_every", type=int, default=1, help="Write results.csv every N runs (0=only at end).")
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=["none", "learned_abs", "learned_abs_drop", "rotary", "rotary_drop"],
        choices=["none", "learned_abs", "learned_abs_drop", "rotary", "rotary_drop"],
        help="Which positional/Drop conditions to run.",
    )
    parser.add_argument("--fineweb_valid_docs", type=int, default=2000, help="FineWeb-Edu validation docs.")
    parser.add_argument("--fineweb_test_docs", type=int, default=2000, help="FineWeb-Edu test docs.")
    parser.add_argument("--fineweb_text_key", type=str, default="text", help="Text field in FineWeb-Edu.")
    parser.add_argument("--prot_text_key", type=str, default="sequence", help="Sequence field in omg_prot50.")
    parser.add_argument("--shuffle_buffer", type=int, default=10000, help="Streaming shuffle buffer size.")
    return parser.parse_args()


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"device must be auto|cpu|cuda, got {device_arg}")


def _fineweb_streams(*, seed: int, shuffle_buffer: int, valid_docs: int, test_docs: int) -> tuple[Iterable[dict], Iterable[dict], Iterable[dict]]:
    base = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    base = base.shuffle(seed=seed, buffer_size=shuffle_buffer)
    valid = base.take(valid_docs)
    test = base.skip(valid_docs).take(test_docs)
    train = base.skip(valid_docs + test_docs)
    return train, valid, test


def _protein_streams(*, seed: int, shuffle_buffer: int) -> tuple[Iterable[dict], Iterable[dict], Iterable[dict]]:
    train = load_dataset("Synthyra/omg_prot50", split="train", streaming=True)
    train = train.shuffle(seed=seed, buffer_size=shuffle_buffer)
    valid = load_dataset("Synthyra/omg_prot50", split="valid", streaming=True)
    test = load_dataset("Synthyra/omg_prot50", split="test", streaming=True)
    return train, valid, test


def _build_loader(
    *,
    dataset: Iterable[dict],
    tokenizer,
    seq_len: int,
    batch_size: int,
    text_key: str,
    repeat: bool,
    mlm_probability: float,
) -> torch.utils.data.DataLoader:
    cfg = StreamConfig(
        seq_len=seq_len,
        batch_size=batch_size,
        text_key=text_key,
        repeat=repeat,
        add_eos=True,
    )
    return build_mlm_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        cfg=cfg,
        mlm_probability=mlm_probability,
        num_workers=0,
    )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _device_from_arg(args.device)
    set_global_seed(0)

    nl_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    prot_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

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
    history_rows: list[dict[str, float | int | str]] = []

    total_runs = len(args.seeds) * len(attention_types) * len(conditions)
    run_idx = 0

    for seed in args.seeds:
        if args.dataset == "nl":
            nl_train_stream, nl_valid_stream, nl_test_stream = _fineweb_streams(
                seed=seed,
                shuffle_buffer=args.shuffle_buffer,
                valid_docs=args.fineweb_valid_docs,
                test_docs=args.fineweb_test_docs,
            )
            prot_train_stream = None
            prot_valid_stream = None
            prot_test_stream = None
            tokenizer = nl_tokenizer
        else:
            nl_train_stream = None
            nl_valid_stream = None
            nl_test_stream = None
            prot_train_stream, prot_valid_stream, prot_test_stream = _protein_streams(
                seed=seed,
                shuffle_buffer=args.shuffle_buffer,
            )
            tokenizer = prot_tokenizer

        for attention_type in attention_types:
            for positional_mode, drop_step in conditions:
                run_idx += 1

                head_size = 128 if attention_type == "dual_triangle" else 64
                if head_size > args.d_model:
                    head_size = args.d_model

                model_cfg = TransformerConfig(
                    vocab_size=int(len(tokenizer)),
                    seq_len=int(args.test_seq_len),
                    d_model=int(args.d_model),
                    n_layers=int(args.n_layers),
                    head_size=int(head_size),
                    d_ff=4 * int(args.d_model),
                    dropout=float(args.dropout),
                    attention_type=attention_type,
                    positional_mode=positional_mode,
                )
                model = init_mlm_model(model_cfg=model_cfg, seed=seed, device=device)

                if args.dataset == "nl":
                    train_loader = _build_loader(
                        dataset=nl_train_stream,
                        tokenizer=nl_tokenizer,
                        seq_len=int(args.train_seq_len),
                        batch_size=int(args.batch_size),
                        text_key=args.fineweb_text_key,
                        repeat=True,
                        mlm_probability=float(args.mlm_prob),
                    )
                    valid_loader = _build_loader(
                        dataset=nl_valid_stream,
                        tokenizer=nl_tokenizer,
                        seq_len=int(args.train_seq_len),
                        batch_size=int(args.batch_size),
                        text_key=args.fineweb_text_key,
                        repeat=False,
                        mlm_probability=float(args.mlm_prob),
                    )
                    test_loader = _build_loader(
                        dataset=nl_test_stream,
                        tokenizer=nl_tokenizer,
                        seq_len=int(args.test_seq_len),
                        batch_size=int(args.batch_size),
                        text_key=args.fineweb_text_key,
                        repeat=False,
                        mlm_probability=float(args.mlm_prob),
                    )
                    phase_name = "nl"
                else:
                    train_loader = _build_loader(
                        dataset=prot_train_stream,
                        tokenizer=prot_tokenizer,
                        seq_len=int(args.train_seq_len),
                        batch_size=int(args.batch_size),
                        text_key=args.prot_text_key,
                        repeat=True,
                        mlm_probability=float(args.mlm_prob),
                    )
                    valid_loader = _build_loader(
                        dataset=prot_valid_stream,
                        tokenizer=prot_tokenizer,
                        seq_len=int(args.train_seq_len),
                        batch_size=int(args.batch_size),
                        text_key=args.prot_text_key,
                        repeat=False,
                        mlm_probability=float(args.mlm_prob),
                    )
                    test_loader = _build_loader(
                        dataset=prot_test_stream,
                        tokenizer=prot_tokenizer,
                        seq_len=int(args.test_seq_len),
                        batch_size=int(args.batch_size),
                        text_key=args.prot_text_key,
                        repeat=False,
                        mlm_probability=float(args.mlm_prob),
                    )
                    phase_name = "protein"

                print(
                    f"[{run_idx}/{total_runs}] seed={seed} attn={attention_type} pos={positional_mode} "
                    f"drop={drop_step} device={device}"
                )

                max_evals = int(args.max_evals_nl) if args.dataset == "nl" else int(args.max_evals_prot)
                train_cfg = MLMTrainConfig(
                    steps_per_eval=int(args.steps),
                    max_evals=max_evals,
                    patience=int(args.patience),
                    warmup_steps=int(args.steps),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    eval_batches=int(args.eval_batches),
                    drop_positions_step=None if drop_step < 0 else int(drop_step),
                    amp=bool(args.amp),
                    mlm_probability=float(args.mlm_prob),
                )

                global_step = 0
                summary, global_step = train_mlm_phase(
                    model=model,
                    train_cfg=train_cfg,
                    device=device,
                    train_iter=iter(train_loader),
                    valid_loader=valid_loader,
                    progress=args.progress,
                    phase_name=phase_name,
                    history_rows=history_rows,
                    start_global_step=global_step,
                )

                test_metrics = eval_mlm(
                    model=model,
                    device=device,
                    loader=test_loader,
                    eval_batches=int(args.eval_batches),
                    amp=bool(args.amp),
                )

                # Clear model from memory
                model.cpu()
                del model
                torch.cuda.empty_cache()

                row = {
                    "seed": seed,
                    "attention_type": attention_type,
                    "positional_mode": positional_mode,
                    "drop_positions_step": -1 if drop_step < 0 else int(drop_step),
                    "d_model": int(args.d_model),
                    "n_layers": int(args.n_layers),
                    "head_size": int(head_size),
                    "train_seq_len": int(args.train_seq_len),
                    "test_seq_len": int(args.test_seq_len),
                }
                if args.dataset == "nl":
                    row.update(
                        {
                            "nl_best_eval_idx": summary["best_eval_idx"],
                            "nl_best_eval_acc": summary["best_eval_acc"],
                            "nl_test_loss": test_metrics["loss"],
                            "nl_test_acc": test_metrics["acc"],
                            "nl_test_f1": test_metrics["f1"],
                            "nl_test_mcc": test_metrics["mcc"],
                        }
                    )
                else:
                    row.update(
                        {
                            "prot_best_eval_idx": summary["best_eval_idx"],
                            "prot_best_eval_acc": summary["best_eval_acc"],
                            "prot_test_loss": test_metrics["loss"],
                            "prot_test_acc": test_metrics["acc"],
                            "prot_test_f1": test_metrics["f1"],
                            "prot_test_mcc": test_metrics["mcc"],
                        }
                    )
                rows.append(row)

                # Attach run metadata to history
                for row in history_rows:
                    if "attention_type" not in row:
                        row["attention_type"] = attention_type
                        row["positional_mode"] = positional_mode
                        row["drop_positions_step"] = -1 if drop_step < 0 else int(drop_step)
                        row["d_model"] = int(args.d_model)
                        row["n_layers"] = int(args.n_layers)
                        row["head_size"] = int(head_size)
                        row["seed"] = seed

                if int(args.flush_every) > 0 and (len(rows) % int(args.flush_every) == 0):
                    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
                    pd.DataFrame(history_rows).to_csv(out_dir / "history.csv", index=False)

    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(history_rows).to_csv(out_dir / "history.csv", index=False)
    plot_all(results_csv=out_dir / "results.csv", history_csv=out_dir / "history.csv", out_dir=out_dir / "plots")
    print(f"Done. Wrote {out_dir / 'results.csv'} and {out_dir / 'history.csv'}")


if __name__ == "__main__":
    main()
