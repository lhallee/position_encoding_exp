from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import torch
from torch.utils.data import IterableDataset, DataLoader


@dataclass(frozen=True)
class StreamConfig:
    seq_len: int
    batch_size: int
    text_key: str
    repeat: bool
    add_eos: bool


def _select_text(sample: dict, text_key: str) -> str:
    if text_key not in sample:
        keys = list(sample.keys())
        raise KeyError(f"Expected key '{text_key}' in sample, got keys={keys}")
    value = sample[text_key]
    if not isinstance(value, str):
        raise ValueError(f"Expected '{text_key}' to be a string, got {type(value)}")
    return value


class StreamingTokenExamples(IterableDataset):
    def __init__(
        self,
        *,
        dataset: Iterable[dict],
        tokenizer,
        cfg: StreamConfig,
    ) -> None:
        super().__init__()
        if cfg.seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {cfg.seq_len}")
        if cfg.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {cfg.batch_size}")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cfg = cfg

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                raise ValueError("Tokenizer must define pad_token_id or eos_token to enable padding")
        self._pad_id = int(tokenizer.pad_token_id)

        if cfg.add_eos:
            if tokenizer.eos_token_id is not None:
                self._eos_id = tokenizer.eos_token_id
            elif tokenizer.sep_token_id is not None:
                self._eos_id = tokenizer.sep_token_id
            else:
                raise ValueError("Tokenizer must define eos_token_id or sep_token_id when add_eos=True")
        else:
            self._eos_id = None

    def _iter_once(self) -> Iterator[torch.Tensor]:
        for sample in self.dataset:
            text = _select_text(sample, self.cfg.text_key)
            ids: list[int] = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.cfg.seq_len,
            )
            if self._eos_id is not None and len(ids) < self.cfg.seq_len:
                ids.append(int(self._eos_id))
            if len(ids) > self.cfg.seq_len:
                ids = ids[: self.cfg.seq_len]

            pad_len = self.cfg.seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self._pad_id] * pad_len
            yield torch.tensor(ids, dtype=torch.long)

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.cfg.repeat:
            while True:
                for item in self._iter_once():
                    yield item
        else:
            yield from self._iter_once()


def _apply_mlm_mask(
    input_ids: torch.Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    special_ids: Sequence[int],
    mlm_probability: float,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids shape (b, t), got {tuple(input_ids.shape)}")
    if mlm_probability <= 0.0 or mlm_probability >= 1.0:
        raise ValueError(f"mlm_probability must be in (0,1), got {mlm_probability}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")

    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, float(mlm_probability), device=labels.device)
    for sid in special_ids:
        probability_matrix = torch.where(labels == int(sid), torch.zeros_like(probability_matrix), probability_matrix)
    probability_matrix = torch.where(attention_mask.bool(), probability_matrix, torch.zeros_like(probability_matrix))
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels = torch.where(masked_indices, labels, torch.full_like(labels, -100))
    labels = torch.where(attention_mask.bool(), labels, torch.full_like(labels, -100))

    # 80% -> [MASK]
    replace_mask = torch.rand(labels.shape, device=labels.device) < 0.8
    indices_replaced = masked_indices & replace_mask
    input_ids = torch.where(indices_replaced, torch.full_like(input_ids, int(mask_token_id)), input_ids)

    # 10% -> random token (of remaining masked)
    random_mask = torch.rand(labels.shape, device=labels.device) < 0.5
    indices_random = masked_indices & (~indices_replaced) & random_mask
    random_tokens = torch.randint(low=0, high=int(vocab_size), size=labels.shape, device=labels.device)
    input_ids = torch.where(indices_random, random_tokens, input_ids)

    # 10% -> keep original
    return input_ids, attention_mask, labels


def build_mlm_dataloader(
    *,
    dataset: Iterable[dict],
    tokenizer,
    cfg: StreamConfig,
    mlm_probability: float,
    num_workers: int = 0,
) -> DataLoader:
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must define mask_token_id for MLM training")
    if len(tokenizer) <= 0:
        raise ValueError("Tokenizer must define a positive vocabulary size for MLM training")

    token_ds = StreamingTokenExamples(dataset=dataset, tokenizer=tokenizer, cfg=cfg)
    special_ids = list(tokenizer.all_special_ids)

    def _collate(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) == 0:
            raise ValueError("Empty batch")
        input_ids = torch.stack(batch, dim=0)
        attention_mask = input_ids != int(tokenizer.pad_token_id)
        return _apply_mlm_mask(
            input_ids,
            mask_token_id=int(tokenizer.mask_token_id),
            vocab_size=int(len(tokenizer)),
            special_ids=special_ids,
            mlm_probability=mlm_probability,
            attention_mask=attention_mask,
        )

    return DataLoader(
        token_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=True,
        persistent_workers=False,
    )


def take_n(it: Iterable[dict], n: int) -> list[dict]:
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    return list(itertools.islice(it, n))
