import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo

from functools import partial
from dataclasses import dataclass
from typing import Optional


Linear = partial(nn.Linear, bias=False)


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    attention_type: str  # "causal" | "bidirectional" | "dual_triangle"
    positional_mode: str  # "none" | "learned_abs" | "rotary"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., d) where d is even; rotate pairs (x0,x1)->(-x1,x0)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    out = torch.stack((-x_odd, x_even), dim=-1)
    return out.flatten(-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (b, h, t, d_rot), cos/sin: (1, 1, t, d_rot)
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, *, seq_len: int, d_rot: int, base: float = 10000.0) -> None:
        super().__init__()
        if d_rot % 2 != 0:
            raise ValueError(f"d_rot must be even, got {d_rot}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        self.seq_len = seq_len
        self.d_rot = d_rot
        self.base = float(base)

        # Buffers are initialized on CPU; we re-cast/move on demand in forward().
        self.register_buffer("cos_cached", torch.empty(1, 1, 0, d_rot), persistent=False)
        self.register_buffer("sin_cached", torch.empty(1, 1, 0, d_rot), persistent=False)
        self.register_buffer("_cache_device", torch.tensor(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_cache_dtype", torch.tensor(0, dtype=torch.int32), persistent=False)

    def _build_cache(self, *, device: torch.device, dtype: torch.dtype) -> None:
        # We compute cos/sin in float32 for stability then cast to dtype.
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.d_rot, 2, device=device, dtype=torch.float32) / float(self.d_rot))
        )  # (d_rot/2,)
        pos = torch.arange(self.seq_len, device=device, dtype=torch.float32)  # (t,)
        freqs = torch.outer(pos, inv_freq)  # (t, d_rot/2)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = cos.repeat_interleave(2, dim=-1)  # (t, d_rot)
        sin = sin.repeat_interleave(2, dim=-1)  # (t, d_rot)
        self.cos_cached = cos.unsqueeze(0).unsqueeze(0).to(dtype=dtype)  # (1,1,t,d_rot)
        self.sin_cached = sin.unsqueeze(0).unsqueeze(0).to(dtype=dtype)  # (1,1,t,d_rot)

    def forward(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cos_cached.shape[2] != self.seq_len:
            self._build_cache(device=device, dtype=dtype)
            return self.cos_cached, self.sin_cached

        # If device/dtype mismatch, rebuild. (Simple + explicit; avoids .get() / hasattr() patterns.)
        if self.cos_cached.device != device or self.cos_cached.dtype != dtype:
            self._build_cache(device=device, dtype=dtype)
        return self.cos_cached, self.sin_cached


class SwiGLUMLP(nn.Module):
    def __init__(self, *, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # SwiGLU: (x W) split -> silu(a) * b -> proj
        self.fc_in = Linear(d_model, 2 * d_ff)
        self.fc_out = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        a, b = torch.chunk(x, 2, dim=-1)
        x = F.silu(a) * b
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


class MultiheadSelfAttention(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, dropout: float, attention_type: str) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        if attention_type not in {"causal", "bidirectional"}:
            raise ValueError(f"attention_type must be causal|bidirectional, got {attention_type}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attention_type = attention_type
        self.dropout = dropout

        self.qkv = Linear(d_model, 3 * d_model)
        self.proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, *, rope_cos: Optional[torch.Tensor], rope_sin: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (b, l, d)
        bsz, t, _ = x.shape
        qkv = self.qkv(x)  # (b, l, 3d)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # (b, l, h, d_head) -> (b, h, l, d_head)
        q = q.view(bsz, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(bsz, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(bsz, t, self.n_heads, self.d_head).transpose(1, 2)

        if rope_cos is not None:
            if rope_sin is None:
                raise ValueError("rope_sin must be provided when rope_cos is provided")
            if self.d_head % 2 != 0:
                raise ValueError(f"RoPE requires even d_head, got d_head={self.d_head}")
            q = _apply_rope(q, rope_cos, rope_sin)
            k = _apply_rope(k, rope_cos, rope_sin)

        # scaled_dot_product_attention returns (b, h, l, d_head)
        if self.attention_type == "causal":
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )

        # (b, h, l, d_head) -> (b, l, d)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, t, self.d_model)
        return self.proj(attn_out)


class DualTriangleAttention(nn.Module):
    """
    Bidirectional attention with separate query-key subspaces for forward/backward directions.
    
    - Upper triangle (j > i): attending to FUTURE positions, uses q_up @ k_up^T
    - Lower triangle (j <= i): attending to PAST/SELF positions, uses q_down @ k_down^T
    
    This allows the model to learn distinct representations for "looking ahead" vs "looking back".
    """
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        d_head = d_model // n_heads
        if d_head % 2 != 0:
            raise ValueError(f"d_head={d_head} must be even (d_model/n_heads must be divisible by 2)")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.half_d = d_head // 2
        self.dropout = dropout

        self.qkv = Linear(d_model, 3 * d_model)
        self.proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, *, rope_cos: Optional[torch.Tensor], rope_sin: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        # Project to Q, K, V and reshape for multi-head attention
        qkv = self.qkv(x)  # (b, l, 3d)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape: (b, l, d) -> (b, h, l, d_head)
        q = q.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        if rope_cos is not None:
            if rope_sin is None:
                raise ValueError("rope_sin must be provided when rope_cos is provided")
            if self.d_head % 2 != 0:
                raise ValueError(f"RoPE requires even d_head, got d_head={self.d_head}")
            q = _apply_rope(q, rope_cos, rope_sin)
            k = _apply_rope(k, rope_cos, rope_sin)

        # Split q and k into halves for upper/lower triangle attention
        # q_up/k_up: for attending to future (j > i)
        # q_down/k_down: for attending to past and self (j <= i)
        q_up, q_down = q[..., : self.half_d], q[..., self.half_d :]
        k_up, k_down = k[..., : self.half_d], k[..., self.half_d :]

        # Compute attention logits for both directions
        # Scale by sqrt(half_d) since each subspace has half the dimensions
        scale = self.half_d**-0.5
        attn_up = torch.matmul(q_up, k_up.transpose(-2, -1)) * scale  # (b, h, l, l)
        attn_down = torch.matmul(q_down, k_down.transpose(-2, -1)) * scale  # (b, h, l, l)

        # Create lower triangle mask (including diagonal): True where j <= i
        lower_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        # Combine: lower triangle (past/self) uses attn_down, upper triangle (future) uses attn_up
        attn_logits = torch.where(lower_mask, attn_down, attn_up)

        # Softmax over keys dimension
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Apply dropout during training
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        # Apply attention weights to values
        attn_out = torch.matmul(attn_weights, v)  # (b, h, l, d_head)

        # Reshape back: (b, h, l, d_head) -> (b, l, d)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, d_ff: int, dropout: float, attention_type: str) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        if attention_type == "dual_triangle":
            self.attn = DualTriangleAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
            self._attn_kind = "dual_triangle"
        else:
            self.attn = MultiheadSelfAttention(
                d_model=d_model, n_heads=n_heads, dropout=dropout, attention_type=attention_type
            )
            self._attn_kind = "sdpa"
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLUMLP(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, rope_cos: Optional[torch.Tensor], rope_sin: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x), rope_cos=rope_cos, rope_sin=rope_sin))
        x = x + self.drop2(self.mlp(self.ln2(x)))
        return x


class PositionProbeTransformer(nn.Module):
    """
    Sequence model that produces a single classification over positions 0..(seq_len-1).

    Pooling head:
      - Compute per-position scores from final hidden states.
      - Softmax over positions to produce weights.
      - Weighted sum -> pooled vector.
      - Linear classifier -> logits over positions.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        if cfg.positional_mode not in {"none", "learned_abs", "rotary"}:
            raise ValueError(f"positional_mode must be none|learned_abs|rotary, got {cfg.positional_mode}")
        if cfg.attention_type not in {"causal", "bidirectional", "dual_triangle"}:
            raise ValueError(f"attention_type must be causal|bidirectional|dual_triangle, got {cfg.attention_type}")

        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if cfg.positional_mode == "learned_abs":
            self.pos_emb = nn.Parameter(torch.zeros(cfg.seq_len, cfg.d_model))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            self.pos_emb = None
        if cfg.positional_mode == "rotary":
            if cfg.d_model % cfg.n_heads != 0:
                raise ValueError(f"d_model={cfg.d_model} must be divisible by n_heads={cfg.n_heads}")
            d_head = cfg.d_model // cfg.n_heads
            if d_head % 2 != 0:
                raise ValueError(f"RoPE requires even d_head, got d_head={d_head}")
            self.rope = RotaryPositionEmbedding(seq_len=cfg.seq_len, d_rot=d_head)
        else:
            self.rope = None

        self.drop_in = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    attention_type=cfg.attention_type,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.pool_score = Linear(cfg.d_model, 1)
        self.classifier = Linear(cfg.d_model, cfg.seq_len)

        self._positions_enabled = True

    def set_positions_enabled(self, enabled: bool) -> None:
        self._positions_enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, l)
        if x.ndim != 2:
            raise ValueError(f"Expected x of shape (b, l), got {tuple(x.shape)}")
        if x.shape[1] != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got {x.shape[1]}")

        # Inputs are expected to be token IDs in [1, vocab_size] inclusive.
        if torch._dynamo.is_compiling():
            # Avoid graph breaks from Tensor.item() while still enforcing the invariant.
            torch._assert(torch.all(x >= 1), "Token IDs must be >= 1")
            torch._assert(
                torch.all(x <= self.cfg.vocab_size),
                f"Token IDs must be <= {self.cfg.vocab_size}",
            )
        else:
            x_min = int(torch.min(x).item())
            x_max = int(torch.max(x).item())
            if x_min < 1 or x_max > self.cfg.vocab_size:
                raise ValueError(
                    f"Token IDs must be in [1, {self.cfg.vocab_size}], got min={x_min} max={x_max}"
                )

        # Map 1..vocab_size -> 0..vocab_size-1 for embedding lookup.
        h = self.tok_emb(x - 1)  # (b, l, d)
        if self.pos_emb is not None and self._positions_enabled:
            h = h + self.pos_emb.unsqueeze(0)  # (1, l, d) -> broadcast

        rope_cos = None
        rope_sin = None
        if self.rope is not None and self._positions_enabled:
            rope_cos, rope_sin = self.rope(device=h.device, dtype=h.dtype)

        h = self.drop_in(h) # (b, l, d)
        for blk in self.blocks:
            h = blk(h, rope_cos=rope_cos, rope_sin=rope_sin)
        h = self.ln_f(h) # (b, l, d)

        # Pool across positions
        scores = self.pool_score(h).squeeze(-1)  # (b, l)
        weights = torch.softmax(scores, dim=1)  # (b, l)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)  # (b, d)
        logits = self.classifier(pooled)  # (b, l)
        return logits

