from __future__ import annotations

import torch


@torch.no_grad()
def sample_batch_argmax_position(
    *,
    batch_size: int,
    seq_len: int,
    vocab_low_inclusive: int,
    vocab_high_inclusive: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Task: predict the (0-indexed) position of the maximum token id in the sequence.

    - Inputs are random integers in [vocab_low_inclusive, vocab_high_inclusive].
    - Labels are argmax positions; ties use the first occurrence (PyTorch argmax behavior).
    """
    x = torch.randint(
        low=vocab_low_inclusive,
        high=vocab_high_inclusive + 1,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    y = torch.argmax(x, dim=1)
    return x, y

