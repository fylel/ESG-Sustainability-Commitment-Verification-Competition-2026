"""
text_clean.py — Text preprocessing ported from the official baseline notebook
(07_roberta_large_raw_only_baseline).

Three stages, all gated by config flags so they can be toggled off:
  1. clean_text_base / normalize_regulatory_text  (USE_TEXT_CLEANING)
  2. mask_company_mentions                         (USE_COMPANY_MASK)
  3. domain tokens added to the tokenizer          (USE_DOMAIN_TOKENS)

Cleaning + masking are applied identically to `data`, `promise_string`
and `evidence_string` so that span-aux alignment (text.find(span)) survives.
"""

import re
import unicodedata
from typing import Optional, Set

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config

# transformers is imported lazily inside build_tokenizer() so that the pure
# text-cleaning helpers can be used (and unit-tested) without the heavy dep.


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — base cleaning + regulatory normalisation
# ──────────────────────────────────────────────────────────────────────

PDF_NOISE_TRANSLATIONS = str.maketrans({
    "\xa0": " ",    # nbsp
    "​": "",   # zero-width space
    "‌": "",
    "‍": "",
    "﻿": "",   # bom
    "　": " ",  # full-width space
})


def clean_text_base(text: Optional[str]) -> str:
    """NFKC normalise, strip PDF noise, collapse whitespace."""
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(PDF_NOISE_TRANSLATIONS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\s+([,.;:!?%％])", r"\1", text)
    return text.strip()


# Standardise framework identifiers so they collapse onto the domain tokens
STANDARD_NORMALIZATION_PATTERNS = [
    (r"(?<![A-Za-z0-9])(GRI)\s*(\d{1,3})(?![A-Za-z0-9])", r"GRI \2"),
    (r"(?<![A-Za-z0-9])(ISO)\s*(\d{4,5}(?:-\d)?)(?![A-Za-z0-9])", r"ISO \2"),
    (r"(?<![A-Za-z0-9])(IFRS)\s*S\s*([12])(?![A-Za-z0-9])", r"IFRS S\2"),
    (r"(?<![A-Za-z0-9])(CO2)\s*e(?![A-Za-z0-9])", r"CO2e"),
    (r"(?<![A-Za-z0-9])(tCO2)\s*e(?![A-Za-z0-9])", r"tCO2e"),
]


def normalize_regulatory_text(text: Optional[str]) -> str:
    """clean_text_base + regulatory term normalisation."""
    text = clean_text_base(text)
    for pattern, repl in STANDARD_NORMALIZATION_PATTERNS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — company masking
# ──────────────────────────────────────────────────────────────────────

COMPANY_MASK_TOKEN = "某公司"
TICKER_MASK_TOKEN = "某代號"

# Suffix patterns (high precision) + verb patterns (broader reach for alias mining).
# Verb patterns are only used for alias MINING (build_company_alias_map), not for
# direct masking, which keeps false-positive masking low.
COMPANY_NAME_PATTERNS = [
    re.compile(r"([一-鿿]{2,10}(?:科技|電子|超商|人壽|金控|水泥|電信|航運|銀行|控股|投控|集團|企業|工業|材料|精密|半導體|電腦|塑膠|化學|製造|儲能))"),
    re.compile(r"([一-鿿]{2,6})致力"),
    re.compile(r"([一-鿿]{2,6})承諾"),
    re.compile(r"([一-鿿]{2,6})重視"),
    re.compile(r"([一-鿿]{2,6})積極"),
    re.compile(r"([一-鿿]{2,6})持續"),
    re.compile(r"([一-鿿]{2,6})為"),
    re.compile(r"([一-鿿]{2,6})以"),
]

COMPANY_NAME_SUFFIXES = (
    "科技", "電子", "超商", "人壽", "金控", "水泥", "電信", "航運", "銀行", "控股", "投控",
    "集團", "企業", "工業", "材料", "精密", "半導體", "電腦", "塑膠", "化學", "製造", "儲能",
)

COMPANY_ALIAS_STOPWORDS = {
    "本公司", "公司", "集團", "我們", "本集團", "董事會", "委員會", "供應商",
    "員工", "產品", "報告書", "近年來", "年度", "未來", "目前", "臺灣",
    "台灣", "全球", "另外", "此外", "因此", "同時",
}


def extract_company_candidates_from_text(text: str) -> list:
    """Extract candidate company names from the first 80 chars using all patterns.

    Only the leading 80 chars are scanned: company names almost always appear
    near the start of a paragraph, and limiting the window reduces false positives
    from verb patterns like 為/以.
    """
    snippet = str(text)[:80]
    candidates = []
    for pattern in COMPANY_NAME_PATTERNS:
        for match in pattern.finditer(snippet):
            candidate = match.group(1).strip("「」『』()（） ")
            if 2 <= len(candidate) <= 10 and candidate not in COMPANY_ALIAS_STOPWORDS:
                candidates.append(candidate)
    return candidates


def is_company_like_candidate(candidate: str, count: int) -> bool:
    """Accept a candidate alias only when it passes count threshold + structural check."""
    if count < 2:
        return False
    if candidate.endswith(COMPANY_NAME_SUFFIXES):
        return True
    if 2 <= len(candidate) <= 4:
        return True
    return False


def build_company_alias_map(samples: list) -> dict:
    """Build a global alias map from a list of sample dicts (aligned with official notebook).

    Groups samples by the 'company' field, mines alias candidates across all
    texts for each company, applies a count≥2 threshold, then assembles the
    final alias list (company name variants + tickers + derived Chinese aliases).

    Pass the result to preprocess_sample / preprocess_text as alias_map so
    every sample in a dataset is masked consistently.
    """
    from collections import Counter, defaultdict

    company_groups: dict = defaultdict(list)
    for s in samples:
        key = str(s.get("company", "")).strip()
        if key and key.lower() != "nan":
            company_groups[key].append(s)

    alias_map: dict = {}
    for company_key, group in company_groups.items():
        counter: Counter = Counter()
        for s in group:
            # Normalize first (same as official: candidate extraction runs on
            # data_normalized_text, not raw text).
            norm = normalize_regulatory_text(str(s.get(config.TEXT_FIELD, "")))
            for candidate in extract_company_candidates_from_text(norm):
                counter[candidate] += 1

        derived = [
            c for c, cnt in counter.most_common(5)
            if is_company_like_candidate(c, cnt)
        ]

        aliases: list = []
        for variant in (company_key, company_key.lower(), company_key.upper()):
            v = variant.strip()
            if v and v.lower() != "nan":
                aliases.append(v)

        tickers: set = set()
        for s in group:
            t = str(s.get("ticker", "")).strip()
            if t and t.lower() != "nan":
                tickers.add(t)
        aliases.extend(sorted(tickers))
        aliases.extend(derived)

        # Dedup, longest first (so longer aliases are replaced before shorter substrings)
        seen: set = set()
        cleaned: list = []
        for item in sorted(aliases, key=len, reverse=True):
            if item and item not in seen:
                seen.add(item)
                cleaned.append(item)

        alias_map[company_key] = cleaned

    return alias_map


def mask_with_aliases(text: str, aliases: Set[str]) -> str:
    """Replace each alias with its mask token (longest first to avoid partials)."""
    if not text:
        return text
    for alias in sorted(aliases, key=len, reverse=True):
        token = TICKER_MASK_TOKEN if alias.isdigit() else COMPANY_MASK_TOKEN
        text = text.replace(alias, token)
    return text


# ──────────────────────────────────────────────────────────────────────
# Combined sample preprocessing (text + spans, kept consistent)
# ──────────────────────────────────────────────────────────────────────

def preprocess_sample(
    text: str,
    promise: str,
    evidence: str,
    sample: dict,
    alias_map: Optional[dict] = None,
):
    """Apply cleaning + masking to text and both span strings identically.

    Returns (text, promise, evidence). When alias_map is provided (built once
    from all training samples via build_company_alias_map), use it to look up
    aliases by company key — this is the official notebook's approach and gives
    better precision than per-sample extraction. Falls back to looking up only
    company/ticker fields from sample when alias_map is None.
    """
    if config.USE_TEXT_CLEANING:
        text = normalize_regulatory_text(text)
        promise = normalize_regulatory_text(promise) if promise else promise
        evidence = normalize_regulatory_text(evidence) if evidence else evidence

    if config.USE_COMPANY_MASK:
        if alias_map is not None:
            company_key = str(sample.get("company", "")).strip()
            aliases: Set[str] = set(alias_map.get(company_key, []))
        else:
            # Fallback: use company/ticker fields only (no verb-pattern mining)
            aliases = set()
            comp = str(sample.get("company", "")).strip()
            if comp and comp.lower() != "nan" and len(comp) >= 2:
                aliases.add(comp)
            ticker = str(sample.get("ticker", "")).strip()
            if ticker and ticker.lower() != "nan":
                aliases.add(ticker)

        if aliases:
            text = mask_with_aliases(text, aliases)
            promise = mask_with_aliases(promise, aliases) if promise else promise
            evidence = mask_with_aliases(evidence, aliases) if evidence else evidence

    return text, promise, evidence


def preprocess_text(
    text: str,
    sample: Optional[dict] = None,
    alias_map: Optional[dict] = None,
) -> str:
    """Clean (+ mask) a single text. Used at inference time in submit.py.

    Pass alias_map (built from training or test samples) for consistent masking.
    """
    if config.USE_TEXT_CLEANING:
        text = normalize_regulatory_text(text)
    if config.USE_COMPANY_MASK and sample is not None:
        if alias_map is not None:
            company_key = str(sample.get("company", "")).strip()
            aliases: Set[str] = set(alias_map.get(company_key, []))
        else:
            aliases = set()
            comp = str(sample.get("company", "")).strip()
            if comp and comp.lower() != "nan" and len(comp) >= 2:
                aliases.add(comp)
            ticker = str(sample.get("ticker", "")).strip()
            if ticker and ticker.lower() != "nan":
                aliases.add(ticker)
        if aliases:
            text = mask_with_aliases(text, aliases)
    return text


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — domain tokens + shared tokenizer
# ──────────────────────────────────────────────────────────────────────

DOMAIN_TOKENS = [
    "IFRS S1", "IFRS S2",
    "GRI 302", "GRI 305", "GRI 306", "GRI 401", "GRI 403",
    "ISO 14064", "ISO 14064-1", "ISO 14067", "ISO 14001", "ISO 45001", "ISO 50001",
    "SASB", "TCFD", "ISSB", "SBTi", "RE100", "CBAM",
    "CO2e", "tCO2e",
]

_TOKENIZER = None
_DOMAIN_TOKEN_PIECE_IDS: dict = {}  # token → [subword_ids] recorded BEFORE adding new tokens


def build_tokenizer():
    """Cached tokenizer with ESG domain tokens added (when enabled).

    Both the dataset and the model must use this so the embedding vocab size
    matches. The model calls resize_token_embeddings(len(build_tokenizer())).
    """
    global _TOKENIZER, _DOMAIN_TOKEN_PIECE_IDS
    if _TOKENIZER is not None:
        return _TOKENIZER
    from transformers import AutoTokenizer  # lazy: keep cleaning import-light
    tok = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL)
    if config.USE_DOMAIN_TOKENS:
        # Record subword pieces BEFORE adding new tokens so we can use them
        # to warm-initialize the new embeddings in init_domain_token_embeddings().
        _DOMAIN_TOKEN_PIECE_IDS = {
            token: tok.convert_tokens_to_ids(tok.tokenize(token))
            for token in DOMAIN_TOKENS
        }
        added = tok.add_tokens(DOMAIN_TOKENS)
        print(f"[tokenizer] added {added} ESG domain tokens (vocab={len(tok)})")
    _TOKENIZER = tok
    return tok


def init_domain_token_embeddings(model) -> None:
    """Warm-initialize new domain token embeddings as the mean of their subword pieces.

    Without this, new tokens start from random noise and need hundreds of steps
    to escape — causing regression when training data is small (~3k samples).

    Call once in ESGMultiTaskModel.__init__ right after resize_token_embeddings().
    Requires build_tokenizer() to have been called first.
    """
    if not config.USE_DOMAIN_TOKENS or not _DOMAIN_TOKEN_PIECE_IDS:
        return
    emb = model.encoder.embeddings.word_embeddings.weight.data
    n_init = 0
    for token, piece_ids in _DOMAIN_TOKEN_PIECE_IDS.items():
        if not piece_ids:
            continue
        new_id = _TOKENIZER.convert_tokens_to_ids(token)
        emb[new_id] = emb[piece_ids].mean(dim=0)
        n_init += 1
    print(f"[domain tokens] warm-initialized {n_init} embeddings from subword averages")
