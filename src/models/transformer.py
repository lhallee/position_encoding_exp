from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    attention_type: str  # "causal" | "bidirectional"
    positional_mode: str  # "none" | "learned_abs"


class MLP(nn.Module):
    def __init__(self, *, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
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

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        bsz, t, _ = x.shape
        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # [B, T, H, Dh] -> [B, H, T, Dh]
        q = q.view(bsz, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(bsz, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(bsz, t, self.n_heads, self.d_head).transpose(1, 2)

        # scaled_dot_product_attention returns [B, H, T, Dh]
        if self.attention_type == "causal":
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )

        # [B, H, T, Dh] -> [B, T, C]
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, t, self.d_model)
        return self.proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, d_ff: int, dropout: float, attention_type: str) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout, attention_type=attention_type
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x)))
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
        if cfg.positional_mode not in {"none", "learned_abs"}:
            raise ValueError(f"positional_mode must be none|learned_abs, got {cfg.positional_mode}")

        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if cfg.positional_mode == "learned_abs":
            self.pos_emb = nn.Parameter(torch.zeros(cfg.seq_len, cfg.d_model))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            self.pos_emb = None

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

        self.pool_score = nn.Linear(cfg.d_model, 1, bias=False)
        self.classifier = nn.Linear(cfg.d_model, cfg.seq_len, bias=True)

        self._positions_enabled = True

    def set_positions_enabled(self, enabled: bool) -> None:
        self._positions_enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        if x.ndim != 2:
            raise ValueError(f"Expected x of shape [B, T], got {tuple(x.shape)}")
        if x.shape[1] != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got {x.shape[1]}")

        # Inputs are expected to be token IDs in [1, vocab_size] inclusive.
        x_min = int(torch.min(x).item())
        x_max = int(torch.max(x).item())
        if x_min < 1 or x_max > self.cfg.vocab_size:
            raise ValueError(
                f"Token IDs must be in [1, {self.cfg.vocab_size}], got min={x_min} max={x_max}"
            )

        # Map 1..vocab_size -> 0..vocab_size-1 for embedding lookup.
        h = self.tok_emb(x - 1)  # [B, T, C]
        if self.pos_emb is not None and self._positions_enabled:
            h = h + self.pos_emb.unsqueeze(0)  # [1, T, C] -> broadcast

        h = self.drop_in(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)

        # Pool across positions
        scores = self.pool_score(h).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=1)  # [B, T]
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)  # [B, C]
        logits = self.classifier(pooled)  # [B, T]
        return logits

