from __future__ import annotations

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from src.models.transformer import PositionProbeTransformer, TransformerConfig, TransformerLM
from src.tasks.mlm_data import StreamConfig, build_mlm_dataloader


def _make_dummy_dataset(texts: list[str]) -> list[dict]:
    return [{"text": t} for t in texts]


def run_experiment1_checks() -> None:
    cfg = TransformerConfig(
        vocab_size=32,
        seq_len=8,
        hidden_size=16,
        n_layers=1,
        head_size=8,
        intermediate_size=32,
        dropout=0.0,
        attention_type="bidirectional",
        positional_mode="learned_abs",
    )
    model = PositionProbeTransformer(cfg)
    x = torch.randint(low=1, high=cfg.vocab_size + 1, size=(2, cfg.seq_len))
    logits = model(x)
    if logits.shape != (2, cfg.seq_len):
        raise ValueError(f"Experiment1 check failed: logits shape {tuple(logits.shape)}")


def run_experiment2_checks() -> None:
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer missing mask_token_id")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer missing pad/eos token")
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
    loader = build_mlm_dataloader(dataset=dataset, tokenizer=tokenizer, cfg=cfg, mlm_probability=0.15)
    input_ids, attention_mask, labels = next(iter(loader))

    if input_ids.shape != (2, cfg.seq_len):
        raise ValueError(f"MLM batch shape mismatch: {tuple(input_ids.shape)}")
    if attention_mask.shape != (2, cfg.seq_len):
        raise ValueError(f"Attention mask shape mismatch: {tuple(attention_mask.shape)}")
    if labels.shape != (2, cfg.seq_len):
        raise ValueError(f"Labels shape mismatch: {tuple(labels.shape)}")

    pad_positions = attention_mask == 0
    if torch.any(labels[pad_positions] != -100):
        raise ValueError("Labels at padded positions must be -100")

    lm_cfg = TransformerConfig(
        vocab_size=len(tokenizer),
        seq_len=cfg.seq_len,
        hidden_size=32,
        n_layers=1,
        head_size=8,
        intermediate_size=64,
        dropout=0.0,
        attention_type="bidirectional",
        positional_mode="learned_abs",
    )
    model = TransformerLM(lm_cfg)
    logits = model(input_ids, attention_mask=attention_mask)
    if logits.shape != (2, cfg.seq_len, len(tokenizer)):
        raise ValueError(f"LM logits shape mismatch: {tuple(logits.shape)}")
