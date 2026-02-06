"""Training loop for Experiment 2: masked language modeling."""

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
    steps_per_eval: int
    max_evals: int
    patience: int
    warmup_steps: int
    batch_size: int
    lr: float
    weight_decay: float
    eval_batches: int
    drop_positions_step: int | None
    amp: bool
    mlm_probability: float
    use_unet: bool = True
    grad_clip: float = 1.0
    muon_lr: float = 0.02


@torch.inference_mode()
def _eval_mlm(
    *,
    model: nn.Module,
    device: torch.device,
    data_iter: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    eval_batches: int,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    use_amp = bool(amp) and (device.type == "cuda")
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

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                )
        else:
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
    amp: bool,
) -> dict[str, float]:
    return _eval_mlm(
        model=model,
        device=device,
        data_iter=iter(loader),
        eval_batches=eval_batches,
        amp=amp,
    )


def init_mlm_model(
    *,
    model_cfg: TransformerConfig,
    seed: int,
    device: torch.device,
    compile_model: bool,
    use_unet: bool = True,
) -> nn.Module:
    set_global_seed(seed)
    if use_unet:
        model = TransformerLMUNet(model_cfg).to(device)
    else:
        model = TransformerLM(model_cfg).to(device)
    if compile_model and sys.platform.startswith("linux"):
        model = torch.compile(model)
    return model


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

    total_steps = train_cfg.max_evals * train_cfg.steps_per_eval
    assert total_steps > 0, f"total_steps must be > 0, got {total_steps}"
    assert train_cfg.warmup_steps > 0, f"warmup_steps must be > 0, got {train_cfg.warmup_steps}"

    use_amp = bool(train_cfg.amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _lr_at_step(step: int) -> float:
        if step < train_cfg.warmup_steps:
            return train_cfg.lr * (float(step + 1) / float(train_cfg.warmup_steps))
        denom = float(max(1, total_steps - train_cfg.warmup_steps))
        progress_val = float(step - train_cfg.warmup_steps) / denom
        progress_val = min(max(progress_val, 0.0), 1.0)
        return train_cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress_val))

    best_metric = -1.0
    best_eval_idx = -1
    best_metrics: dict[str, float] = {}
    bad_evals = 0
    global_step = start_global_step

    for eval_idx in range(train_cfg.max_evals):
        model.train()
        epoch_bar = trange(
            train_cfg.steps_per_eval,
            disable=not progress,
            desc=f"{phase_name} epoch {eval_idx + 1}/{train_cfg.max_evals}",
            leave=False,
        )
        loss_sum = 0.0
        for _ in epoch_bar:
            if train_cfg.drop_positions_step is not None:
                if global_step == train_cfg.drop_positions_step:
                    model.set_positions_enabled(False)

            lr = _lr_at_step(global_step)
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = lr

            input_ids, attention_mask, labels = next(train_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.shape[-1]),
                        labels.view(-1),
                        ignore_index=-100,
                    )
            else:
                logits = model(input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                )

            loss_val = float(loss.item())
            loss_sum += loss_val

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            scaler.scale(loss).backward()

            # Gradient clipping
            if train_cfg.grad_clip > 0:
                for opt in optimizers:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

            for opt in optimizers:
                scaler.step(opt)
            scaler.update()

            global_step += 1
            if progress and (global_step % max(1, train_cfg.steps_per_eval // 4) == 0):
                epoch_bar.set_postfix(loss=loss_val, lr=float(lr))

        train_loss = loss_sum / float(train_cfg.steps_per_eval)
        history_rows.append({
            "phase": phase_name,
            "split": "train",
            "eval_idx": eval_idx,
            "global_step": global_step,
            "loss": train_loss,
        })

        metrics = _eval_mlm(
            model=model,
            device=device,
            data_iter=iter(valid_loader),
            eval_batches=train_cfg.eval_batches,
            amp=train_cfg.amp,
        )
        history_rows.append({
            "phase": phase_name,
            "split": "valid",
            "eval_idx": eval_idx,
            "global_step": global_step,
            "loss": metrics["loss"],
            "acc": metrics["acc"],
            "f1": metrics["f1"],
            "mcc": metrics["mcc"],
        })

        improved = metrics["acc"] > best_metric
        if improved:
            best_metric = metrics["acc"]
            best_eval_idx = eval_idx
            best_metrics = {
                "loss": float(metrics["loss"]),
                "acc": float(metrics["acc"]),
                "f1": float(metrics["f1"]),
                "mcc": float(metrics["mcc"]),
            }
            bad_evals = 0
        else:
            bad_evals += 1

        if bad_evals > train_cfg.patience:
            break

    best_eval_loss = float(best_metrics["loss"]) if "loss" in best_metrics else float("nan")
    best_eval_f1 = float(best_metrics["f1"]) if "f1" in best_metrics else float("nan")
    best_eval_mcc = float(best_metrics["mcc"]) if "mcc" in best_metrics else float("nan")

    summary = {
        "best_eval_idx": best_eval_idx,
        "best_eval_acc": float(best_metric),
        "best_eval_loss": best_eval_loss,
        "best_eval_f1": best_eval_f1,
        "best_eval_mcc": best_eval_mcc,
    }

    return summary, global_step
