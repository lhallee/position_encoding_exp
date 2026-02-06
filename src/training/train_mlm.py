"""Training loop for Experiment 2: masked language modeling.

Uses bfloat16 model casting (like SpeedrunningPLMs) instead of AMP/GradScaler.
LR schedule: linear warmup -> constant -> cosine cooldown.
Always uses last model weights (no early stopping / patience).
"""

from __future__ import annotations

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Iterator
from tqdm import trange

from src.models.transformer import TransformerConfig, TransformerLM, TransformerLMUNet
from src.training.optimizer import build_optimizer
from src.utils.seed import set_global_seed


@dataclass(frozen=True)
class MLMTrainConfig:
    total_steps: int
    warmup_steps: int
    cooldown_steps: int
    eval_every: int
    batch_size: int
    lr: float
    weight_decay: float
    eval_batches: int
    drop_positions_step: int | None
    mlm_probability: float
    use_unet: bool = True
    muon_lr: float = 0.02
    bfloat16: bool = True


@torch.inference_mode()
def _eval_mlm(
    *,
    model: nn.Module,
    device: torch.device,
    data_iter: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    eval_batches: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    eval_steps = 0
    attempts = 0
    max_attempts = eval_batches * 10
    while eval_steps < eval_batches and attempts < max_attempts:
        attempts += 1
        input_ids, attention_mask, labels = next(data_iter)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=-100,
        )

        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        masked_count = int(mask.sum().item())
        if masked_count == 0:
            continue

        eval_steps += 1
        total_loss += float(loss.item())
        total_correct += int(torch.sum((preds == labels) & mask).item())
        total_count += masked_count
        all_preds.extend(preds[mask].flatten().tolist())
        all_labels.extend(labels[mask].flatten().tolist())

    assert total_count > 0, "No masked tokens in evaluation batches; cannot compute metrics."

    acc = float(total_correct) / float(total_count)
    loss_mean = total_loss / float(eval_steps)

    from sklearn.metrics import f1_score, matthews_corrcoef
    f1 = float(f1_score(all_labels, all_preds, average="micro"))
    mcc = float(matthews_corrcoef(all_labels, all_preds))

    return {
        "loss": loss_mean,
        "acc": acc,
        "f1": f1,
        "mcc": mcc,
    }


@torch.inference_mode()
def eval_mlm(
    *,
    model: nn.Module,
    device: torch.device,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    eval_batches: int,
) -> dict[str, float]:
    return _eval_mlm(
        model=model,
        device=device,
        data_iter=iter(loader),
        eval_batches=eval_batches,
    )


def init_mlm_model(
    *,
    model_cfg: TransformerConfig,
    seed: int,
    device: torch.device,
    compile_model: bool,
    use_unet: bool = True,
    bfloat16: bool = True,
) -> nn.Module:
    set_global_seed(seed)
    if use_unet:
        model = TransformerLMUNet(model_cfg).to(device)
    else:
        model = TransformerLM(model_cfg).to(device)

    # Cast to bfloat16 like SpeedrunningPLMs (no AMP/GradScaler needed)
    if bfloat16 and device.type == "cuda":
        model = model.bfloat16()

    if compile_model and sys.platform.startswith("linux"):
        model = torch.compile(model)
    return model


def _lr_schedule(
    step: int,
    *,
    lr: float,
    warmup_steps: int,
    cooldown_steps: int,
    total_steps: int,
) -> float:
    """Linear warmup -> constant -> cosine cooldown to 0."""
    if step < warmup_steps:
        # Linear warmup
        return lr * (float(step + 1) / float(warmup_steps))

    cooldown_start = total_steps - cooldown_steps
    if step >= cooldown_start:
        # Cosine cooldown from lr -> 0
        progress = float(step - cooldown_start) / float(max(1, cooldown_steps))
        progress = min(max(progress, 0.0), 1.0)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Constant LR in the middle
    return lr


def train_mlm_phase(
    *,
    model: nn.Module,
    train_cfg: MLMTrainConfig,
    device: torch.device,
    train_iter: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    valid_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    progress: bool,
    phase_name: str,
    history_rows: list[dict[str, float | int | str]],
    start_global_step: int,
) -> tuple[dict[str, float], int]:
    # Build optimizer: Muon+Adam for UNet, plain AdamW for flat transformer
    if train_cfg.use_unet:
        optimizers = build_optimizer(
            model,
            muon_lr=train_cfg.muon_lr,
            adam_lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
    else:
        optimizers = [torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)]

    assert train_cfg.total_steps > 0, f"total_steps must be > 0, got {train_cfg.total_steps}"
    assert train_cfg.warmup_steps >= 0, f"warmup_steps must be >= 0, got {train_cfg.warmup_steps}"
    assert train_cfg.cooldown_steps >= 0, f"cooldown_steps must be >= 0, got {train_cfg.cooldown_steps}"
    assert train_cfg.warmup_steps + train_cfg.cooldown_steps <= train_cfg.total_steps, (
        f"warmup ({train_cfg.warmup_steps}) + cooldown ({train_cfg.cooldown_steps}) "
        f"exceeds total_steps ({train_cfg.total_steps})"
    )

    # Track last eval metrics (always use last model weights)
    last_metrics: dict[str, float] = {}
    global_step = start_global_step
    eval_idx = 0

    bar = trange(
        train_cfg.total_steps,
        disable=not progress,
        desc=f"{phase_name}",
        leave=False,
    )
    model.train()
    loss_sum = 0.0
    loss_count = 0

    for step_in_phase in bar:
        if train_cfg.drop_positions_step is not None:
            if global_step == train_cfg.drop_positions_step:
                model.set_positions_enabled(False)

        lr = _lr_schedule(
            step_in_phase,
            lr=train_cfg.lr,
            warmup_steps=train_cfg.warmup_steps,
            cooldown_steps=train_cfg.cooldown_steps,
            total_steps=train_cfg.total_steps,
        )
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = lr

        input_ids, attention_mask, labels = next(train_iter)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=-100,
        )

        loss_val = float(loss.item())
        loss_sum += loss_val
        loss_count += 1

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        loss.backward()

        for opt in optimizers:
            opt.step()

        global_step += 1

        if progress and (step_in_phase % max(1, train_cfg.total_steps // 20) == 0):
            avg_loss = loss_sum / max(1, loss_count)
            bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

        # Evaluate periodically
        if train_cfg.eval_every > 0 and (step_in_phase + 1) % train_cfg.eval_every == 0:
            train_loss = loss_sum / max(1, loss_count)
            history_rows.append({
                "phase": phase_name,
                "split": "train",
                "eval_idx": eval_idx,
                "global_step": global_step,
                "loss": train_loss,
            })
            loss_sum = 0.0
            loss_count = 0

            last_metrics = _eval_mlm(
                model=model,
                device=device,
                data_iter=iter(valid_loader),
                eval_batches=train_cfg.eval_batches,
            )
            history_rows.append({
                "phase": phase_name,
                "split": "valid",
                "eval_idx": eval_idx,
                "global_step": global_step,
                "loss": last_metrics["loss"],
                "acc": last_metrics["acc"],
                "f1": last_metrics["f1"],
                "mcc": last_metrics["mcc"],
            })
            eval_idx += 1
            model.train()

    # Final eval at end of training if we haven't just done one
    if loss_count > 0:
        train_loss = loss_sum / max(1, loss_count)
        history_rows.append({
            "phase": phase_name,
            "split": "train",
            "eval_idx": eval_idx,
            "global_step": global_step,
            "loss": train_loss,
        })

    last_metrics = _eval_mlm(
        model=model,
        device=device,
        data_iter=iter(valid_loader),
        eval_batches=train_cfg.eval_batches,
    )
    history_rows.append({
        "phase": phase_name,
        "split": "valid",
        "eval_idx": eval_idx,
        "global_step": global_step,
        "loss": last_metrics["loss"],
        "acc": last_metrics["acc"],
        "f1": last_metrics["f1"],
        "mcc": last_metrics["mcc"],
    })

    # Always use last model weights
    summary = {
        "best_eval_idx": eval_idx,
        "best_eval_acc": float(last_metrics["acc"]),
        "best_eval_loss": float(last_metrics["loss"]),
        "best_eval_f1": float(last_metrics["f1"]),
        "best_eval_mcc": float(last_metrics["mcc"]),
    }

    return summary, global_step
