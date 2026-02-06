"""Sweep runner for Experiment 2: MLM on NLP or protein data."""

import src.entrypoint_setup

import argparse
import itertools
import pandas as pd
import torch
from pathlib import Path
from typing import Iterable
from datasets import load_dataset
from transformers import AutoTokenizer

from src.plotting.plot_mlm import plot_all
from src.training.train_mlm import MLMTrainConfig, eval_mlm, init_mlm_model, train_mlm_phase
from src.models.transformer import TransformerConfig
from src.data.mlm import StreamConfig, build_mlm_dataloader
from src.utils.seed import set_global_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLM sweeps for Experiment 2.")
    parser.add_argument("--dataset", type=str, default="nl", choices=["nl", "protein"], help="Which dataset to run.")
    parser.add_argument("--out_dir", type=str, default="outputs_exp2", help="Output directory.")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile (compiled by default on Linux).")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="LR linear warmup steps.")
    parser.add_argument("--cooldown_steps", type=int, default=1000, help="LR cosine cooldown steps.")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluate every N training steps.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--eval_batches", type=int, default=8, help="Evaluation batches.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11], help="Random seeds.")
    parser.add_argument("--train_seq_len", type=int, default=256, help="Training sequence length.")
    parser.add_argument("--test_seq_len", type=int, default=1024, help="Extended test sequence length.")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers (must be even for UNet).")
    parser.add_argument("--no_progress", action="store_true", help="Disable training progress bars (shown by default).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Adam).")
    parser.add_argument("--muon_lr", type=float, default=0.01, help="Learning rate (Muon).")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--no_bfloat16", action="store_true", help="Disable bfloat16 casting (enabled by default on CUDA).")
    parser.add_argument("--mlm_prob", type=float, default=0.15, help="Masked LM probability.")
    parser.add_argument("--flush_every", type=int, default=1, help="Write results.csv every N runs (0=only at end).")
    parser.add_argument("--no_unet", action="store_true", help="Use flat transformer instead of UNet.")
    parser.add_argument(
        "--conditions", type=str, nargs="+",
        default=["none", "learned_abs", "learned_abs_drop", "rotary", "rotary_drop"],
        choices=["none", "learned_abs", "learned_abs_drop", "rotary", "rotary_drop"],
        help="Which positional/Drop conditions to run.",
    )
    parser.add_argument("--valid_size", type=int, default=2000, help="Validation sample count.")
    parser.add_argument("--test_size", type=int, default=2000, help="Test sample count.")
    parser.add_argument("--fineweb_text_key", type=str, default="text", help="Text field in FineWeb-Edu.")
    parser.add_argument("--prot_text_key", type=str, default="sequence", help="Sequence field in omg_prot50.")
    parser.add_argument("--shuffle_buffer", type=int, default=10000, help="Streaming shuffle buffer size.")
    return parser.parse_args()


def _select_text(sample: dict, text_key: str) -> str:
    assert text_key in sample, f"Expected key '{text_key}' in sample, got keys={list(sample.keys())}"
    value = sample[text_key]
    assert isinstance(value, str), f"Expected '{text_key}' to be a string, got {type(value)}"
    return value


def _take_n(it: Iterable[dict], n: int, *, name: str) -> list[dict]:
    assert n > 0, f"{name} size must be > 0, got {n}"
    items = list(itertools.islice(it, n))
    assert len(items) >= n, f"{name} expected at least {n} items, got {len(items)}"
    return items


def _take_n_filtered_with_consumed(
    it: Iterable[dict], n: int, *, name: str, tokenizer, text_key: str, min_tokens: int,
) -> tuple[list[dict], int]:
    assert n > 0, f"{name} size must be > 0, got {n}"
    assert min_tokens > 0, f"{name} min_tokens must be > 0, got {min_tokens}"
    items: list[dict] = []
    consumed = 0
    for sample in it:
        consumed += 1
        text = _select_text(sample, text_key)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < int(min_tokens):
            continue
        items.append(sample)
        if len(items) >= n:
            break
    assert len(items) >= n, f"{name} expected at least {n} items after filtering, got {len(items)}"
    return items, consumed


def _fineweb_streams(
    *, seed: int, shuffle_buffer: int, valid_size: int, test_size: int,
    tokenizer, text_key: str, test_seq_len: int,
) -> tuple[Iterable[dict], Iterable[dict], Iterable[dict]]:
    base = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    base = base.shuffle(seed=seed, buffer_size=shuffle_buffer)
    valid = _take_n(base.take(valid_size), valid_size, name="FineWeb valid")

    test_candidates = base.skip(valid_size)
    test, consumed = _take_n_filtered_with_consumed(
        test_candidates, test_size, name="FineWeb test",
        tokenizer=tokenizer, text_key=text_key, min_tokens=int(test_seq_len),
    )

    train = base.skip(valid_size + consumed)
    return train, valid, test


def _protein_streams(
    *, seed: int, shuffle_buffer: int, valid_size: int, test_size: int,
    tokenizer, text_key: str, test_seq_len: int,
) -> tuple[Iterable[dict], Iterable[dict], Iterable[dict]]:
    assert valid_size <= 10000, f"Protein valid_size must be <= 10000, got {valid_size}"
    assert test_size <= 10000, f"Protein test_size must be <= 10000, got {test_size}"

    train = load_dataset("Synthyra/omg_prot50", split="train", streaming=True)
    train = train.shuffle(seed=seed, buffer_size=shuffle_buffer)
    valid_stream = load_dataset("Synthyra/omg_prot50", split="valid", streaming=True)
    test_stream = load_dataset("Synthyra/omg_prot50", split="test", streaming=True)

    valid = _take_n(valid_stream, valid_size, name="Protein valid")
    test, _ = _take_n_filtered_with_consumed(
        test_stream, test_size, name="Protein test",
        tokenizer=tokenizer, text_key=text_key, min_tokens=int(test_seq_len),
    )
    return train, valid, test


def _build_loader(
    *, dataset: Iterable[dict], tokenizer, seq_len: int, batch_size: int,
    text_key: str, repeat: bool, mlm_probability: float,
) -> torch.utils.data.DataLoader:
    cfg = StreamConfig(
        seq_len=seq_len, batch_size=batch_size, text_key=text_key, repeat=repeat, add_eos=True,
    )
    return build_mlm_dataloader(
        dataset=dataset, tokenizer=tokenizer, cfg=cfg,
        mlm_probability=mlm_probability, num_workers=2,
    )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(0)

    use_unet = not args.no_unet
    compile_model = not args.no_compile
    show_progress = not args.no_progress
    use_bfloat16 = not args.no_bfloat16

    if args.dataset == "nl":
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        text_key = args.fineweb_text_key
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        text_key = args.prot_text_key

    drop_after = int(2 * args.eval_every)
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
            train_stream, valid_stream, test_stream = _fineweb_streams(
                seed=seed, shuffle_buffer=args.shuffle_buffer,
                valid_size=args.valid_size, test_size=args.test_size,
                tokenizer=tokenizer, text_key=text_key, test_seq_len=int(args.test_seq_len),
            )
            phase_name = "nl"
        else:
            train_stream, valid_stream, test_stream = _protein_streams(
                seed=seed, shuffle_buffer=args.shuffle_buffer,
                valid_size=args.valid_size, test_size=args.test_size,
                tokenizer=tokenizer, text_key=text_key, test_seq_len=int(args.test_seq_len),
            )
            phase_name = "protein"

        train_loader = _build_loader(
            dataset=train_stream, tokenizer=tokenizer, seq_len=int(args.train_seq_len),
            batch_size=int(args.batch_size), text_key=text_key, repeat=True,
            mlm_probability=float(args.mlm_prob),
        )
        valid_loader = _build_loader(
            dataset=valid_stream, tokenizer=tokenizer, seq_len=int(args.train_seq_len),
            batch_size=int(args.batch_size), text_key=text_key, repeat=False,
            mlm_probability=float(args.mlm_prob),
        )
        test_loader = _build_loader(
            dataset=test_stream, tokenizer=tokenizer, seq_len=int(args.test_seq_len),
            batch_size=int(args.batch_size), text_key=text_key, repeat=False,
            mlm_probability=float(args.mlm_prob),
        )

        for attention_type in attention_types:
            for positional_mode, drop_step in conditions:
                run_idx += 1

                head_size = 128 if attention_type == "dual_triangle" else 64
                if head_size > args.hidden_size:
                    head_size = args.hidden_size

                model_cfg = TransformerConfig(
                    vocab_size=int(len(tokenizer)),
                    seq_len=int(args.test_seq_len),
                    hidden_size=int(args.hidden_size),
                    n_layers=int(args.n_layers),
                    head_size=int(head_size),
                    intermediate_size=4 * int(args.hidden_size),
                    dropout=float(args.dropout),
                    attention_type=attention_type,
                    positional_mode=positional_mode,
                )
                model = init_mlm_model(
                    model_cfg=model_cfg, seed=seed, device=device,
                    compile_model=compile_model, use_unet=use_unet,
                    bfloat16=use_bfloat16,
                )

                print(
                    f"[{run_idx}/{total_runs}] seed={seed} attn={attention_type} pos={positional_mode} "
                    f"drop={drop_step} device={device} unet={use_unet} bf16={use_bfloat16}"
                )

                train_cfg = MLMTrainConfig(
                    total_steps=int(args.steps),
                    warmup_steps=int(args.warmup_steps),
                    cooldown_steps=int(args.cooldown_steps),
                    eval_every=int(args.eval_every),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    eval_batches=int(args.eval_batches),
                    drop_positions_step=None if drop_step < 0 else int(drop_step),
                    mlm_probability=float(args.mlm_prob),
                    use_unet=use_unet,
                    muon_lr=float(args.muon_lr),
                    bfloat16=use_bfloat16,
                )

                global_step = 0
                summary, global_step = train_mlm_phase(
                    model=model,
                    train_cfg=train_cfg,
                    device=device,
                    train_iter=iter(train_loader),
                    valid_loader=valid_loader,
                    progress=show_progress,
                    phase_name=phase_name,
                    history_rows=history_rows,
                    start_global_step=global_step,
                )

                test_metrics = eval_mlm(
                    model=model,
                    device=device,
                    loader=test_loader,
                    eval_batches=int(args.eval_batches),
                )

                model.cpu()
                del model
                torch.cuda.empty_cache()

                prefix = "nl" if args.dataset == "nl" else "prot"
                row = {
                    "seed": seed,
                    "attention_type": attention_type,
                    "positional_mode": positional_mode,
                    "drop_positions_step": -1 if drop_step < 0 else int(drop_step),
                    "hidden_size": int(args.hidden_size),
                    "n_layers": int(args.n_layers),
                    "head_size": int(head_size),
                    "train_seq_len": int(args.train_seq_len),
                    "test_seq_len": int(args.test_seq_len),
                    "use_unet": int(use_unet),
                    f"{prefix}_best_eval_idx": summary["best_eval_idx"],
                    f"{prefix}_best_eval_acc": summary["best_eval_acc"],
                    f"{prefix}_best_valid_loss": summary["best_eval_loss"],
                    f"{prefix}_best_valid_acc": summary["best_eval_acc"],
                    f"{prefix}_best_valid_f1": summary["best_eval_f1"],
                    f"{prefix}_best_valid_mcc": summary["best_eval_mcc"],
                    f"{prefix}_test_loss": test_metrics["loss"],
                    f"{prefix}_test_acc": test_metrics["acc"],
                    f"{prefix}_test_f1": test_metrics["f1"],
                    f"{prefix}_test_mcc": test_metrics["mcc"],
                }
                rows.append(row)

                for hr in history_rows:
                    if "attention_type" not in hr:
                        hr["attention_type"] = attention_type
                        hr["positional_mode"] = positional_mode
                        hr["drop_positions_step"] = -1 if drop_step < 0 else int(drop_step)
                        hr["hidden_size"] = int(args.hidden_size)
                        hr["n_layers"] = int(args.n_layers)
                        hr["head_size"] = int(head_size)
                        hr["seed"] = seed

                if int(args.flush_every) > 0 and (len(rows) % int(args.flush_every) == 0):
                    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
                    pd.DataFrame(history_rows).to_csv(out_dir / "history.csv", index=False)

    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(history_rows).to_csv(out_dir / "history.csv", index=False)
    plot_all(results_csv=out_dir / "results.csv", history_csv=out_dir / "history.csv", out_dir=plots_dir)
    print(f"Done. Wrote {out_dir / 'results.csv'} and {out_dir / 'history.csv'}")


if __name__ == "__main__":
    main()
