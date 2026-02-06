"""Custom BPE tokenizer for NLP experiments with small vocabulary."""

from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast


_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

_CACHE_DIR = Path("cache")


def build_nlp_tokenizer(*, vocab_size: int, seed: int, sample_size: int = 50000) -> PreTrainedTokenizerFast:
    """Train a BPE tokenizer on a sample of FineWeb-Edu data with a small vocabulary.

    The tokenizer is cached to disk so it's only trained once per vocab_size.
    """
    assert vocab_size > len(_SPECIAL_TOKENS), (
        f"vocab_size must be > {len(_SPECIAL_TOKENS)} (special tokens), got {vocab_size}"
    )
    assert sample_size > 0, f"sample_size must be > 0, got {sample_size}"

    cache_path = _CACHE_DIR / f"bpe_tokenizer_{vocab_size}"
    if cache_path.exists():
        print(f"Loading cached BPE tokenizer from {cache_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(cache_path),
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        return tokenizer

    print(f"Training BPE tokenizer with vocab_size={vocab_size} from {sample_size} FineWeb-Edu documents...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    texts: list[str] = []
    for sample in ds:
        assert "text" in sample, f"Expected 'text' key in sample, got keys={list(sample.keys())}"
        texts.append(sample["text"])
        if len(texts) >= sample_size:
            break
    assert len(texts) >= sample_size, (
        f"Expected at least {sample_size} documents, got {len(texts)}"
    )

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=_SPECIAL_TOKENS,
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    cache_path.mkdir(parents=True, exist_ok=True)
    wrapped.save_pretrained(str(cache_path))
    print(f"Saved BPE tokenizer ({len(wrapped)} tokens) to {cache_path}")

    return wrapped
