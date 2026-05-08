"""
Tokenizer utilities for VeriPromiseESG4K.

BertTokenizerWrapper — thin wrapper around HuggingFace BertTokenizer
used by the main training pipeline (dataset.py, predict.py).
"""

from typing import List, Optional

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


if __name__ == "__main__":
    bert_tok = BertTokenizerWrapper()
    sample = "公司承諾2025年前達成碳中和目標"
    print(f"Tokens: {bert_tok.tokenize(sample)}")
    ids = bert_tok.encode(sample)
    print(f"IDs:    {ids[:20]}...")
    print(f"Decode: {bert_tok.decode(ids)}")
