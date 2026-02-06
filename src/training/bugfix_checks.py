"""Quick sanity checks for models and data pipelines."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from src.models.transformer import (
    PositionProbeTransformer,
    TransformerConfig,
    TransformerLM,
    TransformerLMUNet,
)
from src.data.mlm import StreamConfig, build_mlm_dataloader


def _make_dummy_dataset(texts: list[str]) -> list[dict]:
    return [{"text": t} for t in texts]


def run_experiment1_checks() -> None:
    print("Running Experiment 1 checks...")

    # Test all attention types
    for attn_type in ["bidirectional", "causal", "dual_triangle"]:
        head_size = 16 if attn_type == "dual_triangle" else 8
        cfg = TransformerConfig(
            vocab_size=32,
            seq_len=8,
            hidden_size=16,
            n_layers=1,
            head_size=head_size,
            intermediate_size=32,
            dropout=0.0,
            attention_type=attn_type,
            positional_mode="learned_abs",
        )
        model = PositionProbeTransformer(cfg)
        x = torch.randint(low=1, high=cfg.vocab_size + 1, size=(2, cfg.seq_len))
        logits = model(x)
        assert logits.shape == (2, cfg.seq_len), (
            f"Exp1 {attn_type} check failed: logits shape {tuple(logits.shape)}"
        )
        print(f"  {attn_type}: OK")

    print("Experiment 1 checks passed.")


def run_experiment2_checks() -> None:
    print("Running Experiment 2 checks...")

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    assert tokenizer.mask_token_id is not None, "Tokenizer missing mask_token_id"
    if tokenizer.pad_token_id is None:
        assert tokenizer.eos_token is not None, "Tokenizer missing pad/eos token"
        tokenizer.pad_token = tokenizer.eos_token

    texts = ["short example", "this is a second example"]
    dataset = _make_dummy_dataset(texts)
    cfg = StreamConfig(
        seq_len=12,
        batch_size=2,
        text_key="text",
        repeat=False,
        add_eos=True,
    )
    loader = build_mlm_dataloader(dataset=dataset, tokenizer=tokenizer, cfg=cfg, mlm_probability=0.15, num_workers=0)
    input_ids, attention_mask, labels = next(iter(loader))

    assert input_ids.shape == (2, cfg.seq_len), f"MLM batch shape mismatch: {tuple(input_ids.shape)}"
    assert attention_mask.shape == (2, cfg.seq_len), f"Attention mask shape mismatch: {tuple(attention_mask.shape)}"
    assert labels.shape == (2, cfg.seq_len), f"Labels shape mismatch: {tuple(labels.shape)}"

    pad_positions = attention_mask == 0
    assert not torch.any(labels[pad_positions] != -100), "Labels at padded positions must be -100"

    # Test flat TransformerLM
    lm_cfg = TransformerConfig(
        vocab_size=len(tokenizer),
        seq_len=cfg.seq_len,
        hidden_size=32,
        n_layers=2,
        head_size=8,
        intermediate_size=64,
        dropout=0.0,
        attention_type="bidirectional",
        positional_mode="learned_abs",
    )
    model = TransformerLM(lm_cfg)
    logits = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (2, cfg.seq_len, len(tokenizer)), (
        f"LM logits shape mismatch: {tuple(logits.shape)}"
    )
    print("  TransformerLM: OK")

    # Test UNet TransformerLMUNet
    unet_cfg = TransformerConfig(
        vocab_size=len(tokenizer),
        seq_len=cfg.seq_len,
        hidden_size=32,
        n_layers=2,
        head_size=8,
        intermediate_size=64,
        dropout=0.0,
        attention_type="bidirectional",
        positional_mode="learned_abs",
    )
    unet_model = TransformerLMUNet(unet_cfg)
    logits = unet_model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (2, cfg.seq_len, len(tokenizer)), (
        f"UNet LM logits shape mismatch: {tuple(logits.shape)}"
    )
    print("  TransformerLMUNet: OK")

    # Test dual triangle attention
    dt_cfg = TransformerConfig(
        vocab_size=len(tokenizer),
        seq_len=cfg.seq_len,
        hidden_size=32,
        n_layers=2,
        head_size=16,
        intermediate_size=64,
        dropout=0.0,
        attention_type="dual_triangle",
        positional_mode="learned_abs",
    )
    dt_model = TransformerLM(dt_cfg)
    logits = dt_model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (2, cfg.seq_len, len(tokenizer)), (
        f"DualTriangle LM logits shape mismatch: {tuple(logits.shape)}"
    )
    print("  DualTriangle attention: OK")

    print("Experiment 2 checks passed.")
