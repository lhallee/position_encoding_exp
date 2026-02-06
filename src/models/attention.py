"""Attention mechanisms using compiled flex_attention.

Provides three attention types:
- Causal (lower triangle)
- Bidirectional (full attention)
- Dual Triangle (head-routed: first N heads attend lower triangle,
                 second N heads attend upper triangle, via a single
                 flex_attention call with a block_mask)
"""

import torch
import torch.nn as nn
from typing import Optional
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    noop_mask,
)

from src.models.components import Linear, apply_rope


# Compiled flex_attention for use in forward passes
_compiled_flex_attention = torch.compile(flex_attention)


# ---------------------------------------------------------------------------
# Mask-mod factories for create_block_mask
# ---------------------------------------------------------------------------

def causal_mask_mod(b, h, q_idx, kv_idx):
    """Causal (autoregressive): query can attend to keys at positions <= q_idx."""
    return q_idx >= kv_idx


def bidirectional_mask_mod(b, h, q_idx, kv_idx):
    """Full bidirectional attention."""
    return q_idx >= 0  # always True


def make_dual_triangle_mask_mod(n_logical_heads: int):
    """Create mask_mod for dual triangle attention.

    Heads 0..n_logical_heads-1 attend to the lower triangle (j <= i, includes diagonal).
    Heads n_logical_heads..2*n_logical_heads-1 attend to the upper triangle (j >= i, includes diagonal).
    Diagonal is included in BOTH triangles for numerical stability (no empty rows).
    """
    def dual_triangle_mask_mod(b, h, q_idx, kv_idx):
        is_down_head = h < n_logical_heads
        lower = q_idx >= kv_idx
        upper = q_idx <= kv_idx
        return (is_down_head & lower) | (~is_down_head & upper)
    return dual_triangle_mask_mod


def make_padding_mask_mod(attention_mask: torch.Tensor):
    """Create mask_mod that masks out padding positions.

    Args:
        attention_mask: (B, L) boolean tensor, True for real tokens.
    """
    def padding_mask_mod(b, h, q_idx, kv_idx):
        return attention_mask[b, q_idx] & attention_mask[b, kv_idx]
    return padding_mask_mod


def compose_masks(*mask_mods):
    """AND-compose multiple mask_mod functions."""
    def composed(b, h, q_idx, kv_idx):
        result = mask_mods[0](b, h, q_idx, kv_idx)
        for mod in mask_mods[1:]:
            result = result & mod(b, h, q_idx, kv_idx)
        return result
    return composed


# ---------------------------------------------------------------------------
# Block mask builders
# ---------------------------------------------------------------------------

@torch.compiler.disable
def build_block_mask(
    *,
    attention_type: str,
    B: int,
    n_heads: int,
    seq_len: int,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
) -> object:
    """Build a BlockMask for flex_attention.

    Args:
        attention_type: "causal" | "bidirectional" | "dual_triangle"
        B: batch size
        n_heads: number of actual heads passed to flex_attention
                 (for dual_triangle this is 2 * n_logical_heads)
        seq_len: sequence length
        device: device for mask computation
        attention_mask: optional (B, L) padding mask (True = real token)
    """
    if attention_type == "causal":
        type_mask = causal_mask_mod
    elif attention_type == "bidirectional":
        type_mask = bidirectional_mask_mod
    elif attention_type == "dual_triangle":
        n_logical = n_heads // 2
        type_mask = make_dual_triangle_mask_mod(n_logical)
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")

    if attention_mask is not None:
        pad_mask = make_padding_mask_mod(attention_mask.bool())
        mask_mod = compose_masks(type_mask, pad_mask)
    else:
        mask_mod = type_mask

    H = n_heads
    block_mask = create_block_mask(
        mask_mod,
        B=B,
        H=H,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )
    return block_mask


# ---------------------------------------------------------------------------
# Attention Modules
# ---------------------------------------------------------------------------

class FlexSelfAttention(nn.Module):
    """Multi-head self attention using compiled flex_attention.

    Supports causal and bidirectional via block_mask.
    """

    def __init__(self, *, hidden_size: int, head_size: int, attention_type: str) -> None:
        super().__init__()
        assert hidden_size % head_size == 0, (
            f"hidden_size={hidden_size} must be divisible by head_size={head_size}"
        )
        assert attention_type in {"causal", "bidirectional"}, (
            f"attention_type must be causal|bidirectional, got {attention_type}"
        )

        self.hidden_size = hidden_size
        self.n_heads = hidden_size // head_size
        self.d_head = head_size
        self.attention_type = attention_type

        self.qkv = Linear(hidden_size, 3 * hidden_size)
        self.proj = Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
        block_mask: object,
    ) -> torch.Tensor:
        bsz, t, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(bsz, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(bsz, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(bsz, t, self.n_heads, self.d_head).transpose(1, 2)

        if rope_cos is not None:
            assert rope_sin is not None, "rope_sin must be provided when rope_cos is provided"
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        attn_out = _compiled_flex_attention(
            q, k, v,
            block_mask=block_mask,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, t, self.hidden_size)
        return self.proj(attn_out)


class DualTriangleFlexAttention(nn.Module):
    """Dual triangle attention using a single compiled flex_attention call.

    Reshapes Q, K, V from (B, n_heads, L, d_head) to (B, 2*n_heads, L, d_head//2).
    First n_heads sub-heads attend to lower triangle (past+self),
    second n_heads sub-heads attend to upper triangle (future+self).
    Each sub-head gets its own softmax normalization.
    """

    def __init__(self, *, hidden_size: int, head_size: int) -> None:
        super().__init__()
        assert hidden_size % head_size == 0, (
            f"hidden_size={hidden_size} must be divisible by head_size={head_size}"
        )
        assert head_size % 2 == 0, (
            f"head_size={head_size} must be even (required for dual triangle splitting)"
        )

        self.hidden_size = hidden_size
        self.n_logical_heads = hidden_size // head_size
        self.d_head = head_size
        self.half_d = head_size // 2
        # Actual number of heads seen by flex_attention
        self.n_actual_heads = 2 * self.n_logical_heads

        self.qkv = Linear(hidden_size, 3 * hidden_size)
        self.proj = Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
        block_mask: object,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # First reshape to logical heads for RoPE
        q = q.view(bsz, seq_len, self.n_logical_heads, self.d_head).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_logical_heads, self.d_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_logical_heads, self.d_head).transpose(1, 2)

        if rope_cos is not None:
            assert rope_sin is not None, "rope_sin must be provided when rope_cos is provided"
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        # Reshape to double heads with half dim: (B, n_logical, L, d_head) -> (B, 2*n_logical, L, d_head//2)
        # Split each head's d_head into two sub-heads of d_head//2
        # First half (down heads): q[..., :half_d], second half (up heads): q[..., half_d:]
        q = q.view(bsz, self.n_logical_heads, seq_len, 2, self.half_d)
        q = q.permute(0, 1, 3, 2, 4).reshape(bsz, self.n_actual_heads, seq_len, self.half_d)

        k = k.view(bsz, self.n_logical_heads, seq_len, 2, self.half_d)
        k = k.permute(0, 1, 3, 2, 4).reshape(bsz, self.n_actual_heads, seq_len, self.half_d)

        v = v.view(bsz, self.n_logical_heads, seq_len, 2, self.half_d)
        v = v.permute(0, 1, 3, 2, 4).reshape(bsz, self.n_actual_heads, seq_len, self.half_d)

        # Single flex_attention call; block_mask routes heads to upper/lower triangles
        attn_out = _compiled_flex_attention(
            q, k, v,
            block_mask=block_mask,
        )

        # Reshape back: (B, 2*n_logical, L, half_d) -> (B, n_logical, L, d_head) -> (B, L, hidden)
        attn_out = attn_out.view(bsz, self.n_logical_heads, 2, seq_len, self.half_d)
        attn_out = attn_out.permute(0, 1, 3, 2, 4).reshape(bsz, self.n_logical_heads, seq_len, self.d_head)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)

        return self.proj(attn_out)
