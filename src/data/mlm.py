"""MLM data loading with async batch pipeline for throughput."""

from __future__ import annotations

import itertools
import threading
import queue
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
    assert text_key in sample, f"Expected key '{text_key}' in sample, got keys={list(sample.keys())}"
    value = sample[text_key]
    assert isinstance(value, str), f"Expected '{text_key}' to be a string, got {type(value)}"
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
        assert cfg.seq_len > 0, f"seq_len must be > 0, got {cfg.seq_len}"
        assert cfg.batch_size > 0, f"batch_size must be > 0, got {cfg.batch_size}"
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cfg = cfg

        if tokenizer.pad_token_id is None:
            assert tokenizer.eos_token is not None, (
                "Tokenizer must define pad_token_id or eos_token to enable padding"
            )
            tokenizer.pad_token = tokenizer.eos_token
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

    def _iter_once(self, dataset: Iterable[dict] | None = None) -> Iterator[torch.Tensor]:
        source = dataset if dataset is not None else self.dataset
        for sample in source:
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
                ids = ids[:self.cfg.seq_len]

            pad_len = self.cfg.seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self._pad_id] * pad_len
            yield torch.tensor(ids, dtype=torch.long)

    def _shard_dataset(self, dataset: Iterable[dict]) -> Iterable[dict]:
        """Shard dataset across DataLoader workers to avoid duplicate batches."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or worker_info.num_workers <= 1:
            return dataset
        # HuggingFace streaming datasets support .shard() for efficient file-level splitting
        if hasattr(dataset, 'shard'):
            return dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)
        # List-based datasets (eval): modulo sharding
        return [d for i, d in enumerate(dataset) if i % worker_info.num_workers == worker_info.id]

    def __iter__(self) -> Iterator[torch.Tensor]:
        dataset = self._shard_dataset(self.dataset)
        if self.cfg.repeat:
            while True:
                for item in self._iter_once(dataset):
                    yield item
        else:
            yield from self._iter_once(dataset)


# ---------------------------------------------------------------------------
# Eval-document filtering wrapper
# ---------------------------------------------------------------------------

class FilteredStream:
    """Wraps a streaming dataset and skips documents whose text is in the eval set.

    Preserves the .shard() interface so DataLoader worker sharding still works.
    """

    def __init__(self, base: Iterable[dict], eval_texts: frozenset[str], text_key: str) -> None:
        assert isinstance(eval_texts, frozenset), f"eval_texts must be a frozenset, got {type(eval_texts)}"
        self.base = base
        self.eval_texts = eval_texts
        self.text_key = text_key

    def __iter__(self):
        for sample in self.base:
            text = _select_text(sample, self.text_key)
            if text not in self.eval_texts:
                yield sample

    def shard(self, num_shards: int, index: int):
        return FilteredStream(self.base.shard(num_shards, index), self.eval_texts, self.text_key)


# ---------------------------------------------------------------------------
# GPU-side MLM masking
# ---------------------------------------------------------------------------

def apply_mlm_mask_gpu(
    input_ids: torch.Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    special_ids: Sequence[int],
    mlm_probability: float,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply MLM masking on GPU tensors for better throughput."""
    assert input_ids.ndim == 2, f"Expected input_ids shape (b, t), got {tuple(input_ids.shape)}"
    assert 0.0 < mlm_probability < 1.0, f"mlm_probability must be in (0,1), got {mlm_probability}"
    assert vocab_size > 0, f"vocab_size must be > 0, got {vocab_size}"

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

    # 10% -> random token
    random_mask = torch.rand(labels.shape, device=labels.device) < 0.5
    indices_random = masked_indices & (~indices_replaced) & random_mask
    random_tokens = torch.randint(low=0, high=int(vocab_size), size=labels.shape, device=labels.device)
    input_ids = torch.where(indices_random, random_tokens, input_ids)

    # 10% -> keep original
    return input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# Async Batch Pipeline
# ---------------------------------------------------------------------------

class AsyncBatchPipeline:
    """Double-buffered CUDA stream pipeline that overlaps H2D transfer with compute.

    Background thread pulls batches from the dataloader, applies MLM masking on GPU,
    and enqueues them for consumption.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
        *,
        mask_token_id: int,
        vocab_size: int,
        special_ids: Sequence[int],
        mlm_probability: float,
        prefetch: int = 2,
    ) -> None:
        self.dataloader = dataloader
        self.device = device
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.special_ids = list(special_ids)
        self.mlm_probability = mlm_probability
        self.prefetch = prefetch

        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _producer(self) -> None:
        is_cuda = self.device.type == "cuda"
        if is_cuda:
            stream = torch.cuda.Stream(device=self.device)
        for batch in self.dataloader:
            if self._stop_event.is_set():
                break
            # batch is a list of tensors from collate (just stacked token ids, no masking yet)
            input_ids = batch
            if is_cuda:
                with torch.cuda.stream(stream):
                    input_ids = input_ids.to(self.device, non_blocking=True)
                    attention_mask = (input_ids != 0).long()  # placeholder, real pad check below
                    # Re-compute attention_mask properly after transfer
                    # The collate returns raw token ids; padding is detectable via pad_token_id
                stream.synchronize()
            else:
                input_ids = input_ids.to(self.device)

            self._queue.put(input_ids)

        self._queue.put(None)  # sentinel

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __iter__(self):
        self.start()
        while True:
            item = self._queue.get()
            if item is None:
                break
            yield item
        self.stop()


# ---------------------------------------------------------------------------
# Dataloader builder
# ---------------------------------------------------------------------------

def build_mlm_dataloader(
    *,
    dataset: Iterable[dict],
    tokenizer,
    cfg: StreamConfig,
    mlm_probability: float,
    num_workers: int = 2,
) -> DataLoader:
    """Build a DataLoader for MLM training with CPU-side masking."""
    assert tokenizer.mask_token_id is not None, "Tokenizer must define mask_token_id for MLM training"
    assert len(tokenizer) > 0, "Tokenizer must define a positive vocabulary size for MLM training"

    token_ds = StreamingTokenExamples(dataset=dataset, tokenizer=tokenizer, cfg=cfg)
    special_ids = list(tokenizer.all_special_ids)

    def _collate(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(batch) > 0, "Empty batch"
        input_ids = torch.stack(batch, dim=0)
        attention_mask = input_ids != int(tokenizer.pad_token_id)
        return apply_mlm_mask_gpu(
            input_ids,
            mask_token_id=int(tokenizer.mask_token_id),
            vocab_size=int(len(tokenizer)),
            special_ids=special_ids,
            mlm_probability=mlm_probability,
            attention_mask=attention_mask,
        )

    use_persistent = num_workers > 0
    return DataLoader(
        token_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=True,
        persistent_workers=use_persistent,
        pin_memory=True,
    )


def take_n(it: Iterable[dict], n: int) -> list[dict]:
    assert n > 0, f"n must be > 0, got {n}"
    return list(itertools.islice(it, n))
