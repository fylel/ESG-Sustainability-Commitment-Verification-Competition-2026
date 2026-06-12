"""
VeriPromiseESG4K dataset utilities.

Loads JSON data, maps labels, builds PyTorch Dataset & DataLoaders.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config
from utils.text_clean import preprocess_sample, build_tokenizer


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

    # commitment=No → evidence/clarity/timeline logically N/A
    if labels["commitment"] == config.COMMITMENT_MAP["No"]:
        for task in ("evidence", "clarity", "timeline"):
            if labels[task] == config.IGNORE_INDEX:
                labels[task] = config.LABEL_MAPS[task]["N/A"]

    # evidence=No → clarity logically N/A
    if labels["evidence"] == config.EVIDENCE_MAP["No"]:
        if labels["clarity"] == config.IGNORE_INDEX:
            labels["clarity"] = config.CLARITY_MAP["N/A"]

    return labels


# ──────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────

_STOP_CHARS = frozenset(
    "的了在是和与或等对为以从到上下中内外于其该此这那个各每某所也都就才又还更最着过吗啊呢吧"
    "，。！？、；：\"""''（）【】《》…—·～ \t\n"
)

_YEAR_RE = re.compile(r'^\d{4}$')

# Phrases that unambiguously signal completion (only used when timeline=already, no target year)
_ALREADY_PHRASES = (
    "已完成", "已達成", "已導入", "已取得", "目前已", "截至本報告年度",
)

# Relative time phrases: label is derived from the sample's timeline class, not fixed
_RELATIVE_TIME_PHRASES = (
    "明年", "後年", "兩年內", "三年內", "五年內",
    "短期", "中期", "長期", "2050淨零", "RE100",
)


def _parse_year(token_text: str) -> int:
    """Return year as int if token is a plausible 4-digit year (1990-2100), else -1."""
    t = token_text.strip()
    if _YEAR_RE.match(t):
        val = int(t)
        if 1990 <= val <= 2100:
            return val
    return -1


def _phrase_chars_in_span(text: str, phrases: tuple, span_char_set: set) -> set:
    """Return char positions of all phrase occurrences that overlap with the span."""
    result: set = set()
    for phrase in phrases:
        start = 0
        while True:
            pos = text.find(phrase, start)
            if pos == -1:
                break
            phrase_chars = set(range(pos, pos + len(phrase)))
            if phrase_chars & span_char_set:
                result.update(phrase_chars)
            start = pos + 1
    return result


def _is_meaningful_token(token_text: str) -> bool:
    """Return True if a token carries semantic weight (not a stop word / punctuation)."""
    t = token_text.strip()
    if not t:
        return False
    if t.isdigit():  # keep years like 2030
        return True
    if len(t) == 1 and t in _STOP_CHARS:
        return False
    if all(c in _STOP_CHARS for c in t):
        return False
    return True


def _find_span_tokens(text: str, span: str, offsets, max_seq_len: int):
    """Return (token_start, token_end) of span within tokenized text, or (-1, -1).

    Uses offset_mapping from HuggingFace fast tokenizer to map character
    positions to token indices. Special/padding tokens have offset (0, 0)
    and are skipped via the e > s check.
    """
    if not span:
        return -1, -1
    char_start = text.find(span)
    if char_start == -1:
        return -1, -1
    char_end = char_start + len(span)  # exclusive

    tok_start = tok_end = -1
    for i, (s, e) in enumerate(offsets):
        if e <= s:  # special or padding token: offset is (0, 0)
            continue
        if s < char_end and e > char_start:  # token overlaps with span
            if tok_start == -1:
                tok_start = i
            tok_end = i

    if tok_start == -1 or tok_end == -1:
        return -1, -1
    return tok_start, tok_end


class ESGDataset(Dataset):
    """
    Each item returns:
        input_ids      – (max_seq_len,)
        attention_mask – (max_seq_len,)
        labels         – stacked int tensor (num_tasks,)
        span_labels    – int tensor [promise_start, promise_end,
                                     evidence_start, evidence_end]
                         -1 means not found / USE_SPAN_AUX is False
    """

    def __init__(
        self,
        samples: List[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = config.MAX_SEQ_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.texts: List[str] = []
        self.labels: List[Dict[str, int]] = []
        self.promise_strings: List[str] = []
        self.evidence_strings: List[str] = []

        self.samples: List[dict] = []
        for s in samples:
            text = normalise_field(s.get(config.TEXT_FIELD, ""))
            if not text:
                continue
            promise = normalise_field(s.get("promise_string", ""))
            evidence = normalise_field(s.get("evidence_string", ""))
            # Clean + company-mask text and span strings together so that
            # text.find(span) used by the span-aux task still resolves.
            text, promise, evidence = preprocess_sample(text, promise, evidence, s)
            if not text:
                continue
            self.samples.append(s)
            self.texts.append(text)
            self.labels.append(encode_labels(s))
            self.promise_strings.append(promise)
            self.evidence_strings.append(evidence)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=config.USE_SPAN_AUX or config.USE_KEYWORD_AUX or config.USE_TEMPORAL_AUX,
        )
        input_ids = enc["input_ids"].squeeze(0)            # (seq_len,)
        attention_mask = enc["attention_mask"].squeeze(0)  # (seq_len,)

        label_tensor = torch.tensor(
            [self.labels[idx][t] for t in config.TASK_NAMES], dtype=torch.long
        )

        offsets = enc["offset_mapping"].squeeze(0).tolist() if "offset_mapping" in enc else None

        # Span labels
        if config.USE_SPAN_AUX and offsets is not None:
            ps, pe = _find_span_tokens(
                self.texts[idx], self.promise_strings[idx], offsets, self.max_seq_len
            )
            es, ee = _find_span_tokens(
                self.texts[idx], self.evidence_strings[idx], offsets, self.max_seq_len
            )
        else:
            ps, pe, es, ee = -1, -1, -1, -1
        span_labels = torch.tensor([ps, pe, es, ee], dtype=torch.long)

        # Keyword labels
        if config.USE_KEYWORD_AUX and offsets is not None:
            keyword_char_set: set = set()
            for span_str in (self.promise_strings[idx], self.evidence_strings[idx]):
                if span_str:
                    cs = self.texts[idx].find(span_str)
                    if cs != -1:
                        keyword_char_set.update(range(cs, cs + len(span_str)))

            if not keyword_char_set:
                keyword_labels = torch.full((self.max_seq_len,), -1, dtype=torch.long)
            else:
                kw = []
                for s, e in offsets:
                    if e <= s:
                        kw.append(-1)
                    elif any(ci in keyword_char_set for ci in range(s, e)):
                        kw.append(1 if _is_meaningful_token(self.texts[idx][s:e]) else 0)
                    else:
                        kw.append(0)
                keyword_labels = torch.tensor(kw, dtype=torch.long)
        else:
            keyword_labels = torch.full((self.max_seq_len,), -1, dtype=torch.long)

        # Temporal labels (near-term=1 / long-term=2 / non-temporal=0 / ignore=-1)
        # span 外 = -1；span 內非時間 token = 0
        if config.USE_TEMPORAL_AUX and offsets is not None:
            timeline_label = self.labels[idx]["timeline"]
            near = config.TIMELINE_NEAR_CLASSES
            far  = config.TIMELINE_FAR_CLASSES
            if timeline_label not in near and timeline_label not in far:
                temporal_labels = torch.full((self.max_seq_len,), -1, dtype=torch.long)
            else:
                target_cls = 1 if timeline_label in near else 2
                text = self.texts[idx]

                span_char_set: set = set()
                for span_str in (self.promise_strings[idx], self.evidence_strings[idx]):
                    if span_str:
                        cs = text.find(span_str)
                        if cs != -1:
                            span_char_set.update(range(cs, cs + len(span_str)))

                # Pass 1: find max year in span
                max_year = -1
                if span_char_set:
                    for s, e in offsets:
                        if e > s and any(ci in span_char_set for ci in range(s, e)):
                            y = _parse_year(text[s:e])
                            if y > max_year:
                                max_year = y

                # Relative time phrase positions in span (label-derived)
                relative_chars = _phrase_chars_in_span(
                    text, _RELATIVE_TIME_PHRASES, span_char_set
                )

                # Already-specific phrases (only when timeline=already AND no target year)
                already_chars: set = set()
                if timeline_label == 0 and max_year == -1:
                    already_chars = _phrase_chars_in_span(
                        text, _ALREADY_PHRASES, span_char_set
                    )

                # Pass 2: assign labels
                tl = []
                for s, e in offsets:
                    if e <= s:
                        tl.append(-1)  # special / padding
                    elif not (span_char_set and any(ci in span_char_set for ci in range(s, e))):
                        tl.append(-1)  # outside span → ignore
                    else:
                        tok_chars = set(range(s, e))
                        y = _parse_year(text[s:e])
                        if y != -1:
                            tl.append(target_cls if y == max_year else 0)
                        elif tok_chars & relative_chars:
                            tl.append(target_cls)
                        elif tok_chars & already_chars:
                            tl.append(1)  # already → near-term
                        else:
                            tl.append(0)

                temporal_labels = torch.tensor(tl, dtype=torch.long)
        else:
            temporal_labels = torch.full((self.max_seq_len,), -1, dtype=torch.long)

        return input_ids, attention_mask, label_tensor, span_labels, keyword_labels, temporal_labels


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
    augment_paths: Optional[List[Path]] = None,
    val_path: Optional[Path] = None,
    merge_val: bool = False,
):
    """Return (train_loader, val_loader, test_loader[, train_ds]).

    val_path: if provided and merge_val=False, use as external validation set
              (all main data goes to training). If merge_val=True, the val
              samples are merged into the training pool and a normal split is
              performed — useful for final submission training.
    augment_paths: augmented samples merged into training data.
    """
    tokenizer = build_tokenizer()  # shared tokenizer (+ ESG domain tokens)
    samples = load_raw_samples(data_path, config.MAX_SAMPLES)

    if augment_paths:
        for aug_path in augment_paths:
            aug = load_raw_samples(Path(aug_path), max_samples=100_000)
            samples.extend(aug)
        print(f"Total samples after augmentation: {len(samples)}")

    if val_path is not None and not merge_val:
        # External val set: train on everything, validate on separate file
        val_samples_raw = load_raw_samples(Path(val_path), max_samples=100_000)
        train_ds = ESGDataset(samples,          tokenizer)
        val_ds   = ESGDataset(val_samples_raw,  tokenizer)
        test_ds  = ESGDataset([],               tokenizer)
        print(f"External val set: {len(val_ds)} samples from {val_path}")
    else:
        if val_path is not None and merge_val:
            val_samples_raw = load_raw_samples(Path(val_path), max_samples=100_000)
            samples.extend(val_samples_raw)
            print(f"merge_val: added {len(val_samples_raw)} val samples to training pool "
                  f"(total {len(samples)})")
        dataset = ESGDataset(samples, tokenizer)
        strat_keys = [
            "_".join(str(lbl[t]) for t in config.TASK_NAMES)
            for lbl in dataset.labels
        ]
        indices = np.arange(len(dataset))

        if test_ratio > 0:
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
                print("Stratified split failed, falling back to random split.")
                train_idx, test_idx = train_test_split(
                    indices, test_size=test_ratio, random_state=seed
                )
                val_ratio_adjusted = val_ratio / (1 - test_ratio)
                train_idx, val_idx = train_test_split(
                    train_idx, test_size=val_ratio_adjusted, random_state=seed
                )
        else:
            test_idx = np.array([], dtype=int)
            try:
                train_idx, val_idx = train_test_split(
                    indices, test_size=val_ratio, random_state=seed, stratify=strat_keys
                )
            except ValueError:
                print("Stratified split failed, falling back to random split.")
                train_idx, val_idx = train_test_split(
                    indices, test_size=val_ratio, random_state=seed
                )

        train_ds = ESGDataset([dataset.samples[i] for i in train_idx], tokenizer)
        val_ds   = ESGDataset([dataset.samples[i] for i in val_idx],   tokenizer)
        test_ds  = ESGDataset([dataset.samples[i] for i in test_idx],  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    if return_train_ds:
        return train_loader, val_loader, test_loader, train_ds
    return train_loader, val_loader, test_loader
