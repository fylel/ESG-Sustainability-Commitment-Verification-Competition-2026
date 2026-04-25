"""
VeriPromiseESG4K dataset utilities.

Loads JSON data, maps labels, builds PyTorch Dataset & DataLoaders.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config


# ──────────────────────────────────────────────────────────────────────
# Raw data helpers
# ──────────────────────────────────────────────────────────────────────

def load_raw_samples(path: Path, max_samples: int = config.MAX_SAMPLES) -> List[dict]:
    """Load JSON / JSONL file and return up to *max_samples* records."""
    raw: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        # try full-file JSON first
        try:
            raw = json.load(f)
            if isinstance(raw, dict):
                # some HF exports wrap the list in a key
                raw = raw.get("data", raw.get("train", [raw]))
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
    return raw[:max_samples]


def normalise_field(value: Optional[str]) -> str:
    """Treat None and empty string uniformly as empty string."""
    if value is None:
        return ""
    return str(value).strip()


def encode_labels(sample: dict) -> Dict[str, int]:
    """Return a dict  task_name -> int label  for one sample."""
    labels: Dict[str, int] = {}
    for task, src_field in config.TASK_SOURCE_FIELDS.items():
        raw_val = normalise_field(sample.get(src_field, ""))
        mapping = config.LABEL_MAPS[task]
        labels[task] = mapping.get(raw_val, config.IGNORE_INDEX)
    return labels


# ──────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────

class ESGDataset(Dataset):
    """
    Each item returns:
        input_ids      – (max_seq_len,)
        attention_mask – (max_seq_len,)
        labels         – dict {task_name: int}   (as a stacked tensor later)
    """

    def __init__(
        self,
        samples: List[dict],
        tokenizer: BertTokenizer,
        max_seq_len: int = config.MAX_SEQ_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.texts: List[str] = []
        self.labels: List[Dict[str, int]] = []

        for s in samples:
            text = normalise_field(s.get(config.TEXT_FIELD, ""))
            if not text:
                continue
            self.texts.append(text)
            self.labels.append(encode_labels(s))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)            # (seq_len,)
        attention_mask = enc["attention_mask"].squeeze(0)  # (seq_len,)

        label_dict = self.labels[idx]
        # stack in canonical task order
        label_tensor = torch.tensor(
            [label_dict[t] for t in config.TASK_NAMES], dtype=torch.long
        )
        return input_ids, attention_mask, label_tensor


# ──────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_path: Path,
    batch_size: int = config.BATCH_SIZE,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.SEED,
    return_train_ds: bool = False,
):
    """Return (train_loader, val_loader, test_loader[, train_ds])."""

    tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL)
    samples = load_raw_samples(data_path, config.MAX_SAMPLES)
    dataset = ESGDataset(samples, tokenizer)

    # stratify key: combine all task labels into one string per sample
    strat_keys = [
        "_".join(str(lbl[t]) for t in config.TASK_NAMES)
        for lbl in dataset.labels
    ]
    indices = np.arange(len(dataset))

    try:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=seed, stratify=strat_keys
        )
        val_ratio_adjusted = val_ratio / (1 - test_ratio)
        strat_keys_train = [strat_keys[i] for i in train_idx]
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_ratio_adjusted, random_state=seed,
            stratify=strat_keys_train
        )
    except ValueError:
        print("Stratified split failed (rare label combos), falling back to random split.")
        train_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=seed
        )
        val_ratio_adjusted = val_ratio / (1 - test_ratio)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_ratio_adjusted, random_state=seed
        )

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    if return_train_ds:
        return train_loader, val_loader, test_loader, train_ds
    return train_loader, val_loader, test_loader
