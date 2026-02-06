"""Training loop for Experiment 1: argmax position probe."""

import src.entrypoint_setup

import math
import sys
import torch
import torch.nn as nn
from tqdm import trange
from dataclasses import dataclass

from src.models.transformer import PositionProbeTransformer, TransformerConfig
from src.data.argmax import sample_batch_argmax_position
from src.utils.seed import set_global_seed


@dataclass(frozen=True)
class TrainConfig:
    steps_per_eval: int
    max_evals: int
    patience: int
    warmup_steps: int
    batch_size: int
    lr: float
    weight_decay: float
    eval_batches: int
    drop_positions_step: int | None
    label_mode: str  # "true" | "random"


@torch.inference_mode()
def evaluate_accuracy(
    *,
    model: nn.Module,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    vocab_low_inclusive: int,
    vocab_high_inclusive: int,
    label_mode: str,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for _ in range(eval_batches):
        x, y = sample_batch_argmax_position(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_low_inclusive=vocab_low_inclusive,
            vocab_high_inclusive=vocab_high_inclusive,
            device=device,
        )
        if label_mode == "random":
            y = torch.randint(low=0, high=seq_len, size=y.shape, device=device, dtype=torch.long)
        elif label_mode == "true":
            pass
        else:
            raise ValueError(f"label_mode must be true|random, got {label_mode}")
        
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += int(torch.sum(pred == y).item())
        total += int(y.numel())
    return correct / total


def train_one(
    *,
    model_cfg: TransformerConfig,
    train_cfg: TrainConfig,
    seed: int,
    device: torch.device,
    vocab_low_inclusive: int,
    vocab_high_inclusive: int,
    progress: bool,
    wandb_run=None,
) -> dict[str, float | int | str]:
    set_global_seed(seed)

    model = PositionProbeTransformer(model_cfg).to(device)

    if sys.platform.startswith("linux"):
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    total_steps = train_cfg.max_evals * train_cfg.steps_per_eval
    assert total_steps > 0, f"total_steps must be > 0, got {total_steps}"
    assert train_cfg.warmup_steps > 0, f"warmup_steps must be > 0, got {train_cfg.warmup_steps}"

    def _lr_at_step(step: int) -> float:
        if step < train_cfg.warmup_steps:
            return train_cfg.lr * (float(step + 1) / float(train_cfg.warmup_steps))
        denom = float(max(1, total_steps - train_cfg.warmup_steps))
        progress_val = float(step - train_cfg.warmup_steps) / denom
        progress_val = min(max(progress_val, 0.0), 1.0)
        return train_cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress_val))

    best_acc = -1.0
    best_eval_idx = -1
    bad_evals = 0
    last_acc = -1.0

    global_step = 0
    for eval_idx in range(train_cfg.max_evals):
        model.train()
        epoch_bar = trange(
            train_cfg.steps_per_eval,
            disable=not progress,
            desc=f"epoch {eval_idx + 1}/{train_cfg.max_evals}",
            leave=False,
        )
        last_loss = None
        for _ in epoch_bar:
            if train_cfg.drop_positions_step is not None:
                if global_step == train_cfg.drop_positions_step:
                    model.set_positions_enabled(False)

            lr = _lr_at_step(global_step)
            for group in opt.param_groups:
                group["lr"] = lr

            x, y = sample_batch_argmax_position(
                batch_size=train_cfg.batch_size,
                seq_len=model_cfg.seq_len,
                vocab_low_inclusive=vocab_low_inclusive,
                vocab_high_inclusive=vocab_high_inclusive,
                device=device,
            )
            if train_cfg.label_mode == "random":
                y = torch.randint(low=0, high=model_cfg.seq_len, size=y.shape, device=device, dtype=torch.long)
            elif train_cfg.label_mode == "true":
                pass
            else:
                raise ValueError(f"label_mode must be true|random, got {train_cfg.label_mode}")

            logits = model(x)
            loss = loss_fn(logits, y)
            last_loss = float(loss.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            global_step += 1

            if wandb_run is not None:
                wandb_run.log({"train/loss": last_loss, "train/lr": float(lr)}, step=global_step)

            if progress and (global_step % max(1, train_cfg.steps_per_eval // 4) == 0):
                epoch_bar.set_postfix(loss=last_loss, lr=float(lr))

        acc = evaluate_accuracy(
            model=model,
            device=device,
            seq_len=model_cfg.seq_len,
            batch_size=train_cfg.batch_size,
            eval_batches=train_cfg.eval_batches,
            vocab_low_inclusive=vocab_low_inclusive,
            vocab_high_inclusive=vocab_high_inclusive,
            label_mode=train_cfg.label_mode,
        )
        last_acc = acc

        if wandb_run is not None:
            wandb_run.log({
                "eval/accuracy": float(acc),
                "eval/best_accuracy": float(best_acc) if acc <= best_acc else float(acc),
                "eval/epoch": eval_idx,
            }, step=global_step)

        improved = acc > best_acc
        if improved:
            best_acc = acc
            best_eval_idx = eval_idx
            bad_evals = 0
        else:
            bad_evals += 1

        if train_cfg.drop_positions_step is None and acc >= 1.0:
            best_acc = 1.0
            best_eval_idx = eval_idx
            last_acc = 1.0
            break

        if progress:
            epoch_bar.set_postfix(
                loss=last_loss,
                lr=float(_lr_at_step(max(0, global_step - 1))),
                acc=float(acc),
                best=float(best_acc),
                bad=bad_evals,
            )

        if bad_evals > train_cfg.patience:
            break

    model.cpu()
    del model
    torch.cuda.empty_cache()

    return {
        "seed": seed,
        "seq_len": model_cfg.seq_len,
        "vocab_size": model_cfg.vocab_size,
        "attention_type": model_cfg.attention_type,
        "positional_mode": model_cfg.positional_mode,
        "drop_positions_step": -1 if train_cfg.drop_positions_step is None else train_cfg.drop_positions_step,
        "label_mode": train_cfg.label_mode,
        "n_layers": model_cfg.n_layers,
        "hidden_size": model_cfg.hidden_size,
        "head_size": model_cfg.head_size,
        "intermediate_size": model_cfg.intermediate_size,
        "steps_per_eval": train_cfg.steps_per_eval,
        "max_evals": train_cfg.max_evals,
        "patience": train_cfg.patience,
        "warmup_steps": train_cfg.warmup_steps,
        "trained_steps": global_step,
        "best_eval_idx": best_eval_idx,
        "batch_size": train_cfg.batch_size,
        "lr": train_cfg.lr,
        "weight_decay": train_cfg.weight_decay,
        "eval_batches": train_cfg.eval_batches,
        "eval_acc": float(last_acc),
        "best_eval_acc": float(best_acc),
    }
