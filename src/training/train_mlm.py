"""Training loop for Experiment 2: masked language modeling.

Trains until a target number of non-pad tokens have been consumed.
Uses bfloat16 model casting (like SpeedrunningPLMs) instead of AMP/GradScaler.
LR schedule: linear warmup -> constant -> cosine cooldown (all token-based).
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
from tqdm import tqdm

from src.models.transformer import TransformerConfig, TransformerLM, TransformerLMUNet
from src.training.optimizer import build_optimizer
from src.utils.seed import set_global_seed


@dataclass(frozen=True)
class MLMTrainConfig:
    total_tokens: int
    warmup_tokens: int
    cooldown_tokens: int
    eval_every: int
    batch_size: int
    lr: float
    weight_decay: float
    drop_positions_tokens: int | None
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
    progress: bool = False,
) -> dict[str, float]:
    """Evaluate on the entire dataset provided by data_iter."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    eval_steps = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    eval_bar = tqdm(desc="eval", leave=False, disable=not progress)
    for input_ids, attention_mask, labels in data_iter:
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
        eval_bar.update(1)

    eval_bar.close()
    assert total_count > 0, "No masked tokens in evaluation data; cannot compute metrics."

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
    progress: bool = False,
) -> dict[str, float]:
    return _eval_mlm(
        model=model,
        device=device,
        data_iter=iter(loader),
        progress=progress,
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
    tokens_seen: int,
    *,
    lr: float,
    warmup_tokens: int,
    cooldown_tokens: int,
    total_tokens: int,
) -> float:
    """Linear warmup -> constant -> cosine cooldown to 0 (token-based)."""
    if tokens_seen < warmup_tokens:
        # Linear warmup
        return lr * (float(tokens_seen) / float(max(1, warmup_tokens)))

    cooldown_start = total_tokens - cooldown_tokens
    if tokens_seen >= cooldown_start:
        # Cosine cooldown from lr -> 0
        progress = float(tokens_seen - cooldown_start) / float(max(1, cooldown_tokens))
        progress = min(max(progress, 0.0), 1.0)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Constant LR in the middle
    return lr


def _run_validation(
    *,
    model: nn.Module,
    device: torch.device,
    valid_short_loader: Iterable,
    valid_long_loader: Iterable,
    progress: bool,
    phase_name: str,
    eval_idx: int,
    global_step: int,
    tokens_seen: int,
    history_rows: list[dict[str, float | int | str]],
    wandb_run,
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate on both valid_short and valid_long, log to history/wandb."""
    short_metrics = _eval_mlm(
        model=model, device=device,
        data_iter=iter(valid_short_loader),
        progress=progress,
    )
    history_rows.append({
        "phase": phase_name,
        "split": "valid_short",
        "eval_idx": eval_idx,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "loss": short_metrics["loss"],
        "acc": short_metrics["acc"],
        "f1": short_metrics["f1"],
        "mcc": short_metrics["mcc"],
    })
    if wandb_run is not None:
        wandb_run.log({
            "valid_short/loss": short_metrics["loss"],
            "valid_short/accuracy": short_metrics["acc"],
            "valid_short/f1": short_metrics["f1"],
            "valid_short/mcc": short_metrics["mcc"],
        }, step=global_step)

    long_metrics = _eval_mlm(
        model=model, device=device,
        data_iter=iter(valid_long_loader),
        progress=progress,
    )
    history_rows.append({
        "phase": phase_name,
        "split": "valid_long",
        "eval_idx": eval_idx,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "loss": long_metrics["loss"],
        "acc": long_metrics["acc"],
        "f1": long_metrics["f1"],
        "mcc": long_metrics["mcc"],
    })
    if wandb_run is not None:
        wandb_run.log({
            "valid_long/loss": long_metrics["loss"],
            "valid_long/accuracy": long_metrics["acc"],
            "valid_long/f1": long_metrics["f1"],
            "valid_long/mcc": long_metrics["mcc"],
        }, step=global_step)

    return short_metrics, long_metrics


def train_mlm_phase(
    *,
    model: nn.Module,
    train_cfg: MLMTrainConfig,
    device: torch.device,
    train_iter: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    valid_short_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    valid_long_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    progress: bool,
    phase_name: str,
    history_rows: list[dict[str, float | int | str]],
    start_global_step: int,
    start_tokens_seen: int = 0,
    wandb_run=None,
) -> tuple[dict[str, float], int, int]:
    """Train until total_tokens non-pad tokens have been consumed.

    Returns (summary_dict, final_global_step, final_tokens_seen).
    """
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

    assert train_cfg.total_tokens > 0, f"total_tokens must be > 0, got {train_cfg.total_tokens}"
    assert train_cfg.warmup_tokens >= 0, f"warmup_tokens must be >= 0, got {train_cfg.warmup_tokens}"
    assert train_cfg.cooldown_tokens >= 0, f"cooldown_tokens must be >= 0, got {train_cfg.cooldown_tokens}"
    assert train_cfg.warmup_tokens + train_cfg.cooldown_tokens <= train_cfg.total_tokens, (
        f"warmup ({train_cfg.warmup_tokens}) + cooldown ({train_cfg.cooldown_tokens}) "
        f"exceeds total_tokens ({train_cfg.total_tokens})"
    )

    # Track last eval metrics (always use last model weights)
    last_short_metrics: dict[str, float] = {}
    last_long_metrics: dict[str, float] = {}
    global_step = start_global_step
    tokens_seen = start_tokens_seen
    eval_idx = 0
    positions_dropped = False

    bar = tqdm(
        total=train_cfg.total_tokens,
        initial=start_tokens_seen,
        unit="tok",
        unit_scale=True,
        disable=not progress,
        desc=f"{phase_name}",
        leave=False,
    )
    model.train()
    loss_sum = 0.0
    loss_count = 0
    wandb_loss_sum = 0.0
    wandb_loss_count = 0
    step_in_phase = 0

    while tokens_seen < train_cfg.total_tokens:
        if train_cfg.drop_positions_tokens is not None and not positions_dropped:
            if tokens_seen >= train_cfg.drop_positions_tokens:
                model.set_positions_enabled(False)
                positions_dropped = True

        lr = _lr_schedule(
            tokens_seen,
            lr=train_cfg.lr,
            warmup_tokens=train_cfg.warmup_tokens,
            cooldown_tokens=train_cfg.cooldown_tokens,
            total_tokens=train_cfg.total_tokens,
        )
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = lr

        input_ids, attention_mask, labels = next(train_iter)
        # Count non-pad tokens on CPU before device transfer (avoids GPU sync)
        batch_tokens = int(attention_mask.sum().item())
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
        wandb_loss_sum += loss_val
        wandb_loss_count += 1

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        loss.backward()

        for opt in optimizers:
            opt.step()

        tokens_seen += batch_tokens
        global_step += 1
        step_in_phase += 1
        bar.update(batch_tokens)

        if wandb_run is not None and wandb_loss_count >= 100:
            wandb_run.log({"train/loss": wandb_loss_sum / wandb_loss_count, "train/lr": lr}, step=global_step)
            wandb_loss_sum = 0.0
            wandb_loss_count = 0

        if progress and (step_in_phase % 100 == 0):
            avg_loss = loss_sum / max(1, loss_count)
            bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

        # Evaluate periodically (eval_every is still step-based)
        if train_cfg.eval_every > 0 and step_in_phase % train_cfg.eval_every == 0:
            train_loss = loss_sum / max(1, loss_count)
            history_rows.append({
                "phase": phase_name,
                "split": "train",
                "eval_idx": eval_idx,
                "global_step": global_step,
                "tokens_seen": tokens_seen,
                "loss": train_loss,
            })

            if wandb_run is not None:
                wandb_run.log({"train/loss": train_loss, "train/lr": lr}, step=global_step)

            loss_sum = 0.0
            loss_count = 0
            wandb_loss_sum = 0.0
            wandb_loss_count = 0

            last_short_metrics, last_long_metrics = _run_validation(
                model=model, device=device,
                valid_short_loader=valid_short_loader,
                valid_long_loader=valid_long_loader,
                progress=progress,
                phase_name=phase_name,
                eval_idx=eval_idx,
                global_step=global_step,
                tokens_seen=tokens_seen,
                history_rows=history_rows,
                wandb_run=wandb_run,
            )

            eval_idx += 1
            model.train()

    bar.close()

    # Final eval at end of training if we haven't just done one
    if loss_count > 0:
        train_loss = loss_sum / max(1, loss_count)
        history_rows.append({
            "phase": phase_name,
            "split": "train",
            "eval_idx": eval_idx,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "loss": train_loss,
        })

        if wandb_run is not None:
            wandb_run.log({"train/loss": train_loss, "train/lr": lr}, step=global_step)

    last_short_metrics, last_long_metrics = _run_validation(
        model=model, device=device,
        valid_short_loader=valid_short_loader,
        valid_long_loader=valid_long_loader,
        progress=progress,
        phase_name=phase_name,
        eval_idx=eval_idx,
        global_step=global_step,
        tokens_seen=tokens_seen,
        history_rows=history_rows,
        wandb_run=wandb_run,
    )

    # Always use last model weights
    summary = {
        "best_eval_idx": eval_idx,
        "best_eval_short_acc": float(last_short_metrics["acc"]),
        "best_eval_short_loss": float(last_short_metrics["loss"]),
        "best_eval_short_f1": float(last_short_metrics["f1"]),
        "best_eval_short_mcc": float(last_short_metrics["mcc"]),
        "best_eval_long_acc": float(last_long_metrics["acc"]),
        "best_eval_long_loss": float(last_long_metrics["loss"]),
        "best_eval_long_f1": float(last_long_metrics["f1"]),
        "best_eval_long_mcc": float(last_long_metrics["mcc"]),
        "total_tokens_seen": tokens_seen,
    }

    return summary, global_step, tokens_seen
