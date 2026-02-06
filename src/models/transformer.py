"""Transformer models: PositionProbeTransformer, TransformerLM, TransformerLMUNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from dataclasses import dataclass
from typing import Optional

from src.models.components import (
    Linear,
    RotaryPositionEmbedding,
    SwiGLUMLP,
    apply_rope,
)
from src.models.attention import (
    FlexSelfAttention,
    DualTriangleFlexAttention,
    build_block_mask,
)


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    seq_len: int
    hidden_size: int
    n_layers: int
    head_size: int
    intermediate_size: int
    dropout: float
    attention_type: str  # "causal" | "bidirectional" | "dual_triangle"
    positional_mode: str  # "none" | "learned_abs" | "rotary"


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        head_size: int,
        intermediate_size: int,
        dropout: float,
        attention_type: str,
        unet: bool = False,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attention_type = attention_type
        self.unet = unet

        if attention_type == "dual_triangle":
            self.attn = DualTriangleFlexAttention(
                hidden_size=hidden_size,
                head_size=head_size,
            )
        else:
            self.attn = FlexSelfAttention(
                hidden_size=hidden_size,
                head_size=head_size,
                attention_type=attention_type,
            )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = SwiGLUMLP(hidden_size=hidden_size, intermediate_size=intermediate_size, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

        if unet:
            # Lambda mixing for UNet: x = lambda[0]*x + lambda[1]*x0
            self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
            # Value embedding mixing in attention
            self.v_lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
        block_mask: object,
        x0: Optional[torch.Tensor] = None,
        vi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.unet and x0 is not None:
            x = self.lambdas[0] * x + self.lambdas[1] * x0

        x = x + self.drop1(
            self.attn(
                self.ln1(x),
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                block_mask=block_mask,
            )
        )
        x = x + self.drop2(self.mlp(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Value Embedding for UNet
# ---------------------------------------------------------------------------

class ValueEmbedding(nn.Module):
    """Per-layer value embeddings that mirror encoder/decoder structure."""

    def __init__(self, vocab_size: int, hidden_size: int, n_encoder_layers: int) -> None:
        super().__init__()
        self.embed = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_size)
            for _ in range(n_encoder_layers)
        ])

    def forward(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        """Returns list of value embeddings, mirrored for decoder."""
        ve = [emb(input_ids) for emb in self.embed]
        # Mirror for decoder: [...encoder_ve, ...reversed(encoder_ve)]
        return ve + list(reversed(ve))


# ---------------------------------------------------------------------------
# Position Probe Transformer (Experiment 1)
# ---------------------------------------------------------------------------

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
        assert cfg.positional_mode in {"none", "learned_abs", "rotary"}, (
            f"positional_mode must be none|learned_abs|rotary, got {cfg.positional_mode}"
        )
        assert cfg.attention_type in {"causal", "bidirectional", "dual_triangle"}, (
            f"attention_type must be causal|bidirectional|dual_triangle, got {cfg.attention_type}"
        )

        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        if cfg.positional_mode == "learned_abs":
            self.pos_emb = nn.Parameter(torch.zeros(cfg.seq_len, cfg.hidden_size))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            self.pos_emb = None

        if cfg.positional_mode == "rotary":
            assert cfg.hidden_size % cfg.head_size == 0, (
                f"hidden_size={cfg.hidden_size} must be divisible by head_size={cfg.head_size}"
            )
            assert cfg.head_size % 2 == 0, f"RoPE requires even head_size, got head_size={cfg.head_size}"
            self.rope = RotaryPositionEmbedding(seq_len=cfg.seq_len, d_rot=cfg.head_size)
        else:
            self.rope = None

        self.drop_in = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=cfg.hidden_size,
                head_size=cfg.head_size,
                intermediate_size=cfg.intermediate_size,
                dropout=cfg.dropout,
                attention_type=cfg.attention_type,
            )
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.hidden_size)

        self.pool_score = Linear(cfg.hidden_size, 1)
        self.classifier = Linear(cfg.hidden_size, cfg.seq_len)

        self._positions_enabled = True

        # Cache block mask (fixed seq_len, no padding for this model)
        self._cached_block_mask = None
        self._cached_device = None

    def _n_flex_heads(self) -> int:
        n_heads = self.cfg.hidden_size // self.cfg.head_size
        if self.cfg.attention_type == "dual_triangle":
            return 2 * n_heads
        return n_heads

    def set_positions_enabled(self, enabled: bool) -> None:
        self._positions_enabled = enabled

    def _get_block_mask(self, device: torch.device) -> object:
        if self._cached_block_mask is not None and self._cached_device == device:
            return self._cached_block_mask
        self._cached_block_mask = build_block_mask(
            attention_type=self.cfg.attention_type,
            B=1,  # broadcast across batch
            n_heads=self._n_flex_heads(),
            seq_len=self.cfg.seq_len,
            device=device,
        )
        self._cached_device = device
        return self._cached_block_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"Expected x of shape (b, l), got {tuple(x.shape)}"
        assert x.shape[1] == self.cfg.seq_len, f"Expected seq_len={self.cfg.seq_len}, got {x.shape[1]}"

        if torch.compiler.is_compiling():
            torch._assert(torch.all(x >= 1), "Token IDs must be >= 1")
            torch._assert(
                torch.all(x <= self.cfg.vocab_size),
                f"Token IDs must be <= {self.cfg.vocab_size}",
            )
        else:
            x_min = int(torch.min(x).item())
            x_max = int(torch.max(x).item())
            assert 1 <= x_min and x_max <= self.cfg.vocab_size, (
                f"Token IDs must be in [1, {self.cfg.vocab_size}], got min={x_min} max={x_max}"
            )

        h = self.tok_emb(x - 1)
        if self.pos_emb is not None and self._positions_enabled:
            h = h + self.pos_emb.unsqueeze(0)

        rope_cos = None
        rope_sin = None
        if self.rope is not None and self._positions_enabled:
            rope_cos, rope_sin = self.rope(device=h.device, dtype=h.dtype)

        h = self.drop_in(h)

        block_mask = self._get_block_mask(h.device)
        for blk in self.blocks:
            h = blk(h, rope_cos=rope_cos, rope_sin=rope_sin, block_mask=block_mask)
        h = self.ln_f(h)

        scores = self.pool_score(h).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)
        return logits


# ---------------------------------------------------------------------------
# Transformer LM (Flat, for MLM - Experiment 2 baseline)
# ---------------------------------------------------------------------------

class TransformerLM(nn.Module):
    """Flat transformer for per-token language modeling."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        assert cfg.positional_mode in {"none", "learned_abs", "rotary"}
        assert cfg.attention_type in {"causal", "bidirectional", "dual_triangle"}

        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        if cfg.positional_mode == "learned_abs":
            self.pos_emb = nn.Parameter(torch.zeros(cfg.seq_len, cfg.hidden_size))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            self.pos_emb = None

        if cfg.positional_mode == "rotary":
            assert cfg.hidden_size % cfg.head_size == 0
            assert cfg.head_size % 2 == 0, f"RoPE requires even head_size, got {cfg.head_size}"
            self.rope = RotaryPositionEmbedding(seq_len=cfg.seq_len, d_rot=cfg.head_size)
        else:
            self.rope = None

        self.drop_in = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=cfg.hidden_size,
                head_size=cfg.head_size,
                intermediate_size=cfg.intermediate_size,
                dropout=cfg.dropout,
                attention_type=cfg.attention_type,
            )
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=True),
        )
        self._positions_enabled = True

    def _n_flex_heads(self) -> int:
        n_heads = self.cfg.hidden_size // self.cfg.head_size
        if self.cfg.attention_type == "dual_triangle":
            return 2 * n_heads
        return n_heads

    def set_positions_enabled(self, enabled: bool) -> None:
        self._positions_enabled = enabled

    def forward(self, x: torch.Tensor, *, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        assert x.ndim == 2, f"Expected x of shape (b, t), got {tuple(x.shape)}"
        assert x.shape[1] <= self.cfg.seq_len, f"Expected seq_len <= {self.cfg.seq_len}, got {x.shape[1]}"

        bsz, t = x.shape
        h = self.tok_emb(x)
        if self.pos_emb is not None and self._positions_enabled:
            h = h + self.pos_emb[:t].unsqueeze(0)

        rope_cos = None
        rope_sin = None
        if self.rope is not None and self._positions_enabled:
            rope_cos, rope_sin = self.rope(device=h.device, dtype=h.dtype)
            rope_cos = rope_cos[:, :, :t, :]
            rope_sin = rope_sin[:, :, :t, :]

        h = self.drop_in(h)

        block_mask = build_block_mask(
            attention_type=self.cfg.attention_type,
            B=bsz,
            n_heads=self._n_flex_heads(),
            seq_len=t,
            device=h.device,
            attention_mask=attention_mask,
        )

        for blk in self.blocks:
            h = blk(h, rope_cos=rope_cos, rope_sin=rope_sin, block_mask=block_mask)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits


# ---------------------------------------------------------------------------
# UNet Transformer LM (Experiment 2)
# ---------------------------------------------------------------------------

class TransformerLMUNet(nn.Module):
    """UNet-style transformer for per-token language modeling.

    Architecture:
    - Encoder half: layers save skip connections
    - Decoder half: layers add back skip connections with learnable weights
    - Value embeddings: per-layer embeddings mixed via lambdas
    - x0 mixing: original input residual mixed into each layer
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        assert cfg.positional_mode in {"none", "learned_abs", "rotary"}
        assert cfg.attention_type in {"causal", "bidirectional", "dual_triangle"}
        assert cfg.n_layers % 2 == 0, f"UNet requires even n_layers, got {cfg.n_layers}"

        self.cfg = cfg
        self.n_encoder_layers = cfg.n_layers // 2
        self.n_decoder_layers = cfg.n_layers // 2

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        if cfg.positional_mode == "learned_abs":
            self.pos_emb = nn.Parameter(torch.zeros(cfg.seq_len, cfg.hidden_size))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            self.pos_emb = None

        if cfg.positional_mode == "rotary":
            assert cfg.hidden_size % cfg.head_size == 0
            assert cfg.head_size % 2 == 0, f"RoPE requires even head_size, got {cfg.head_size}"
            self.rope = RotaryPositionEmbedding(seq_len=cfg.seq_len, d_rot=cfg.head_size)
        else:
            self.rope = None

        self.drop_in = nn.Dropout(cfg.dropout)

        # All layers are UNet-style blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=cfg.hidden_size,
                head_size=cfg.head_size,
                intermediate_size=cfg.intermediate_size,
                dropout=cfg.dropout,
                attention_type=cfg.attention_type,
                unet=True,
            )
            for _ in range(cfg.n_layers)
        ])

        # Skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.n_decoder_layers))

        # Value embeddings
        self.value_embeds = ValueEmbedding(cfg.vocab_size, cfg.hidden_size, self.n_encoder_layers)

        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=True),
        )
        self._positions_enabled = True

    def _n_flex_heads(self) -> int:
        n_heads = self.cfg.hidden_size // self.cfg.head_size
        if self.cfg.attention_type == "dual_triangle":
            return 2 * n_heads
        return n_heads

    def set_positions_enabled(self, enabled: bool) -> None:
        self._positions_enabled = enabled

    def forward(self, x: torch.Tensor, *, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        assert x.ndim == 2, f"Expected x of shape (b, t), got {tuple(x.shape)}"
        assert x.shape[1] <= self.cfg.seq_len, f"Expected seq_len <= {self.cfg.seq_len}, got {x.shape[1]}"

        bsz, t = x.shape

        # Value embeddings from raw input_ids
        ve = self.value_embeds(x)
        ve_enc = ve[:self.n_encoder_layers]
        ve_dec = ve[self.n_encoder_layers:]

        h = self.tok_emb(x)
        if self.pos_emb is not None and self._positions_enabled:
            h = h + self.pos_emb[:t].unsqueeze(0)

        rope_cos = None
        rope_sin = None
        if self.rope is not None and self._positions_enabled:
            rope_cos, rope_sin = self.rope(device=h.device, dtype=h.dtype)
            rope_cos = rope_cos[:, :, :t, :]
            rope_sin = rope_sin[:, :, :t, :]

        h = self.drop_in(h)
        x0 = h.clone()

        block_mask = build_block_mask(
            attention_type=self.cfg.attention_type,
            B=bsz,
            n_heads=self._n_flex_heads(),
            seq_len=t,
            device=h.device,
            attention_mask=attention_mask,
        )

        # Encoder
        skip_connections = []
        for i in range(self.n_encoder_layers):
            h = self.blocks[i](
                h,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                block_mask=block_mask,
                x0=x0,
                vi=ve_enc[i],
            )
            skip_connections.append(h)

        # Decoder
        for i in range(self.n_decoder_layers):
            h = h + self.skip_weights[i] * skip_connections.pop()
            h = self.blocks[self.n_encoder_layers + i](
                h,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                block_mask=block_mask,
                x0=x0,
                vi=ve_dec[i],
            )

        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits
