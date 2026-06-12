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

# High-precision: a Chinese name immediately followed by a company-type suffix.
# (The notebook's lower-precision verb patterns are only for alias mining; using
#  them to mask directly would corrupt ordinary text, so we keep this one.)
_COMPANY_SUFFIXES = (
    "科技電子超商人壽金控控股投控水泥電信航運銀行集團企業工業"
    "材料精密半導體電腦塑膠化學製造儲能"
)
COMPANY_NAME_PATTERN = re.compile(
    r"[一-鿿]{2,10}(?:"
    + "|".join(
        ("科技", "電子", "超商", "人壽", "金控", "控股", "投控", "水泥",
         "電信", "航運", "銀行", "集團", "企業", "工業", "材料", "精密",
         "半導體", "電腦", "塑膠", "化學", "製造", "儲能")
    )
    + ")"
)

COMPANY_ALIAS_STOPWORDS = {
    "本公司", "公司", "集團", "我們", "本集團", "董事會", "委員會", "供應商",
    "員工", "產品", "報告書", "近年來", "年度", "未來", "目前", "臺灣",
    "台灣", "全球", "另外", "此外", "因此", "同時",
}


def extract_company_aliases(text: str, sample: dict) -> Set[str]:
    """Per-sample alias set: known company/ticker fields + Chinese names in text.

    Derived from the (already cleaned) text so the same set can mask text and
    both span strings consistently. Works on unseen test companies too.
    """
    aliases: Set[str] = set()

    comp = str(sample.get("company", "")).strip()
    if comp and not comp.isdigit() and len(comp) >= 2:
        aliases.add(comp)

    ticker = str(sample.get("ticker", "")).strip()
    if ticker.isdigit() and len(ticker) >= 3:
        aliases.add(ticker)

    for name in COMPANY_NAME_PATTERN.findall(text):
        if len(name) >= 3 and name not in COMPANY_ALIAS_STOPWORDS:
            aliases.add(name)

    return aliases


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

def preprocess_sample(text: str, promise: str, evidence: str, sample: dict):
    """Apply cleaning + masking to text and both span strings identically.

    Returns (text, promise, evidence). Aliases are derived once from the
    cleaned full text and applied to all three so that text.find(span) used by
    the span-aux task still resolves after masking.
    """
    if config.USE_TEXT_CLEANING:
        text = normalize_regulatory_text(text)
        promise = normalize_regulatory_text(promise) if promise else promise
        evidence = normalize_regulatory_text(evidence) if evidence else evidence

    if config.USE_COMPANY_MASK:
        aliases = extract_company_aliases(text, sample)
        if aliases:
            text = mask_with_aliases(text, aliases)
            promise = mask_with_aliases(promise, aliases) if promise else promise
            evidence = mask_with_aliases(evidence, aliases) if evidence else evidence

    return text, promise, evidence


def preprocess_text(text: str, sample: Optional[dict] = None) -> str:
    """Clean (+ mask) a single text. Used at inference time in submit.py."""
    if config.USE_TEXT_CLEANING:
        text = normalize_regulatory_text(text)
    if config.USE_COMPANY_MASK and sample is not None:
        aliases = extract_company_aliases(text, sample)
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


def build_tokenizer():
    """Cached tokenizer with ESG domain tokens added (when enabled).

    Both the dataset and the model must use this so the embedding vocab size
    matches. The model calls resize_token_embeddings(len(build_tokenizer())).
    """
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    from transformers import AutoTokenizer  # lazy: keep cleaning import-light
    tok = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL)
    if config.USE_DOMAIN_TOKENS:
        added = tok.add_tokens(DOMAIN_TOKENS)
        print(f"[tokenizer] added {added} ESG domain tokens (vocab={len(tok)})")
    _TOKENIZER = tok
    return tok
