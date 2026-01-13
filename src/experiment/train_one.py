from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import trange

from src.models.transformer import PositionProbeTransformer, TransformerConfig
from src.tasks.argmax_position import sample_batch_argmax_position
from src.utils.seed import set_global_seed


@dataclass(frozen=True)
class TrainConfig:
    steps: int
    batch_size: int
    lr: float
    weight_decay: float
    eval_batches: int
    drop_positions_step: int | None  # if set, disable positions at this step and keep training


@torch.no_grad()
def evaluate_accuracy(
    *,
    model: nn.Module,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    vocab_low_inclusive: int,
    vocab_high_inclusive: int,
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
) -> dict[str, float | int | str]:
    set_global_seed(seed)
    model = PositionProbeTransformer(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    iterator = trange(train_cfg.steps, disable=not progress, desc="train", leave=False)
    for step in iterator:
        if train_cfg.drop_positions_step is not None:
            if step == train_cfg.drop_positions_step:
                model.set_positions_enabled(False)

        x, y = sample_batch_argmax_position(
            batch_size=train_cfg.batch_size,
            seq_len=model_cfg.seq_len,
            vocab_low_inclusive=vocab_low_inclusive,
            vocab_high_inclusive=vocab_high_inclusive,
            device=device,
        )

        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if progress and (step + 1) % max(1, train_cfg.steps // 5) == 0:
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                acc = torch.mean((pred == y).float()).item()
            iterator.set_postfix(loss=float(loss.item()), acc=float(acc))

    acc = evaluate_accuracy(
        model=model,
        device=device,
        seq_len=model_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        eval_batches=train_cfg.eval_batches,
        vocab_low_inclusive=vocab_low_inclusive,
        vocab_high_inclusive=vocab_high_inclusive,
    )

    return {
        "seed": seed,
        "seq_len": model_cfg.seq_len,
        "vocab_size": model_cfg.vocab_size,
        "attention_type": model_cfg.attention_type,
        "positional_mode": model_cfg.positional_mode,
        "drop_positions_step": -1 if train_cfg.drop_positions_step is None else train_cfg.drop_positions_step,
        "n_layers": model_cfg.n_layers,
        "d_model": model_cfg.d_model,
        "n_heads": model_cfg.n_heads,
        "d_ff": model_cfg.d_ff,
        "steps": train_cfg.steps,
        "batch_size": train_cfg.batch_size,
        "lr": train_cfg.lr,
        "weight_decay": train_cfg.weight_decay,
        "eval_batches": train_cfg.eval_batches,
        "eval_acc": float(acc),
    }

