"""Shared model components: Linear, RoPE, SwiGLUMLP."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


Linear = partial(nn.Linear, bias=False)


# ---------------------------------------------------------------------------
# Rotary Position Embedding
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., d) where d is even; rotate pairs (x0,x1)->(-x1,x0)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    out = torch.stack((-x_odd, x_even), dim=-1)
    return out.flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to tensor x.

    Args:
        x: (b, h, t, d_rot)
        cos, sin: (1, 1, t, d_rot)
    """
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, *, seq_len: int, d_rot: int, base: float = 10000.0) -> None:
        super().__init__()
        assert d_rot % 2 == 0, f"d_rot must be even, got {d_rot}"
        assert seq_len > 0, f"seq_len must be > 0, got {seq_len}"
        self.seq_len = seq_len
        self.d_rot = d_rot
        self.base = float(base)

        self.register_buffer("cos_cached", torch.empty(1, 1, 0, d_rot), persistent=False)
        self.register_buffer("sin_cached", torch.empty(1, 1, 0, d_rot), persistent=False)

    def _build_cache(self, *, device: torch.device, dtype: torch.dtype) -> None:
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.d_rot, 2, device=device, dtype=torch.float32) / float(self.d_rot))
        )
        pos = torch.arange(self.seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        self.cos_cached = cos.unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        self.sin_cached = sin.unsqueeze(0).unsqueeze(0).to(dtype=dtype)

    def forward(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cos_cached.shape[2] != self.seq_len:
            self._build_cache(device=device, dtype=dtype)
            return self.cos_cached, self.sin_cached

        if self.cos_cached.device != device or self.cos_cached.dtype != dtype:
            self._build_cache(device=device, dtype=dtype)
        return self.cos_cached, self.sin_cached


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    def __init__(self, *, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.fc_in = Linear(hidden_size, 2 * intermediate_size)
        self.fc_out = Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        a, b = torch.chunk(x, 2, dim=-1)
        x = F.silu(a) * b
        x = self.dropout(x)
        x = self.fc_out(x)
        return x
