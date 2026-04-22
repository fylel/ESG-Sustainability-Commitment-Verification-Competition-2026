"""
Tokenizer utilities for VeriPromiseESG4K.

This module provides:
  1. BertTokenizerWrapper — thin wrapper around HuggingFace BertTokenizer
     used by the main training pipeline (dataset.py, predict.py).
  2. ChineseCharTokenizer — character-level tokenizer for Traditional/Simplified
     Chinese text, useful for vocab analysis, data exploration, and fallback.

The original translation-project BaseTokenizer architecture is preserved
for compatibility and extended with ESG-specific helpers.
"""

from pathlib import Path
from typing import List, Optional

from tqdm import tqdm
from transformers import BertTokenizer

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config


# ──────────────────────────────────────────────────────────────────────
# 1. BERT wrapper (primary tokenizer for the pipeline)
# ──────────────────────────────────────────────────────────────────────

class BertTokenizerWrapper:
    """
    Convenience wrapper so the rest of the codebase can call
    `tokenizer.encode(text)` without touching HuggingFace details.
    """

    def __init__(self, pretrained: str = config.PRETRAINED_MODEL,
                 max_len: int = config.MAX_SEQ_LEN):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.max_len = max_len
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_index = self.tokenizer.pad_token_id
        self.cls_token_index = self.tokenizer.cls_token_id
        self.sep_token_index = self.tokenizer.sep_token_id

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def encode(self, text: str, add_special_tokens: bool = True):
        return self.tokenizer.encode(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=add_special_tokens,
        )

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids,
                                     skip_special_tokens=skip_special_tokens)


# ──────────────────────────────────────────────────────────────────────
# 2. Character-level tokenizer (preserved from original project)
# ──────────────────────────────────────────────────────────────────────

class BaseTokenizer:
    """Base tokenizer with vocab build / load from file."""

    unk_token = "<unk>"
    pad_token = "<pad>"
    sos_token = "<sos>"
    eos_token = "<eos>"

    def __init__(self, vocab_list: List[str]):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: idx for idx, word in enumerate(vocab_list)}
        self.index2word = {idx: word for idx, word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        raise NotImplementedError

    def encode(self, text: str, add_sos_eos: bool = False) -> List[int]:
        """Tokenize text → token IDs, optionally wrapping with <sos>/<eos>."""
        tokens = self.tokenize(text)
        if add_sos_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        return [self.word2index.get(t, self.unk_token_index) for t in tokens]

    def decode(self, indexes: List[int]) -> str:
        """Convert token IDs back to a string."""
        tokens = [self.index2word.get(i, self.unk_token) for i in indexes]
        # filter special tokens
        tokens = [t for t in tokens
                  if t not in (self.pad_token, self.sos_token, self.eos_token)]
        return "".join(tokens)

    # ── Vocab build / load ────────────────────────────────────────────

    @classmethod
    def build_vocab(cls, sentences: List[str], vocab_path) -> None:
        """Build vocab from sentences and save to file (one token per line)."""
        vocab_path = Path(vocab_path)
        vocab_set: set = set()
        for sentence in tqdm(sentences, desc="構建詞表"):
            vocab_set.update(cls.tokenize(sentence))
        vocab_list = (
            [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token]
            + [t for t in sorted(vocab_set) if t.strip()]
        )
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab_list))
        print(f"詞表已保存 → {vocab_path}  (大小: {len(vocab_list)})")

    @classmethod
    def from_vocab(cls, vocab_path):
        """Instantiate from a saved vocab file."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)


class ChineseCharTokenizer(BaseTokenizer):
    """Character-level Chinese tokenizer (one char = one token)."""

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        return list(text)


# ──────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # BERT tokenizer
    bert_tok = BertTokenizerWrapper()
    sample = "公司承諾2025年前達成碳中和目標"
    print("=== BertTokenizerWrapper ===")
    print(f"Text:   {sample}")
    print(f"Tokens: {bert_tok.tokenize(sample)}")
    ids = bert_tok.encode(sample)
    print(f"IDs:    {ids[:20]}...")
    print(f"Decode: {bert_tok.decode(ids)}")

    # Char-level tokenizer
    print("\n=== ChineseCharTokenizer ===")
    print(f"Tokens: {ChineseCharTokenizer.tokenize(sample)}")
