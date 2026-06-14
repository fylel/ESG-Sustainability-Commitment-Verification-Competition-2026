"""
text_clean.py — Text preprocessing ported from the official baseline notebook
(07_roberta_large_raw_only_baseline).

Three stages, all gated by config flags so they can be toggled off:
  1. clean_text_base / normalize_regulatory_text  (USE_TEXT_CLEANING)
  2. mask_company_mentions                         (USE_COMPANY_MASK)
  3. domain tokens added to the tokenizer          (USE_DOMAIN_TOKENS)

Plus hybrid features (USE_HYBRID_FEATURES):
  - feat_C / feat_E: rule-based (computed live for every sample)
  - feat_A / feat_B / feat_D: loaded from GPT-4o CSV; default 0 for synthetic data
  Injected as a 5-token prefix: [A=0][B=0][C=1][D=0][E=1]

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
    hybrid_map: Optional[dict] = None,
):
    """Apply cleaning + masking to text and both span strings identically.

    Returns (text, promise, evidence). When alias_map is provided (built once
    from all training samples via build_company_alias_map), use it to look up
    aliases by company key. hybrid_map (built via build_hybrid_feature_map)
    injects a 5-feature prefix to the main text only (not spans).
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

    if config.USE_HYBRID_FEATURES:
        sample_id = str(sample.get("id", ""))
        feats = _resolve_hybrid_feats(text, sample_id, hybrid_map)
        prefix = build_hybrid_prefix(feats)
        text = prefix + text

    return text, promise, evidence


def preprocess_text(
    text: str,
    sample: Optional[dict] = None,
    alias_map: Optional[dict] = None,
    hybrid_map: Optional[dict] = None,
) -> str:
    """Clean (+ mask) a single text. Used at inference time in submit.py.

    Pass alias_map (built from training or test samples) for consistent masking.
    Pass hybrid_map (built via build_hybrid_feature_map) for hybrid feature prefix.
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
    if config.USE_HYBRID_FEATURES:
        sample_id = str(sample.get("id", "")) if sample else ""
        feats = _resolve_hybrid_feats(text, sample_id, hybrid_map)
        prefix = build_hybrid_prefix(feats)
        text = prefix + text
    return text


# ──────────────────────────────────────────────────────────────────────
# Hybrid Features — rule-based (C/E) + GPT-4o CSV lookup (A/B/D)
# ──────────────────────────────────────────────────────────────────────

# feat_C: measurable signal (rule-based) — ported from official notebook Phase B-2
_MEASURE_VALUE = r"(?:\d+(?:[,.]\d+)*(?:\.\d+)?|[零〇一二兩三四五六七八九十百千萬億兆壹貳參肆伍陸柒捌玖拾佰仟]+)"
_MEASURE_CHINESE_NUM = r"""
    數[十百千萬億兆]+ |
    [零〇一二兩三四五六七八九壹貳參肆伍陸柒捌玖]+[十百千萬億兆拾佰仟]+[零〇一二兩三四五六七八九十百千萬億兆壹貳參肆伍陸柒捌玖拾佰仟]* |
    [十拾][一二兩三四五六七八九壹貳參肆伍陸柒捌玖]+
"""
_MEASURE_RATIO = r"""
    減半 |
    [零〇一二兩三四五六七八九十百千萬億兆]+成 |
    百分之[零〇一二兩三四五六七八九十百千萬億兆\d]+ |
    千分之[零〇一二兩三四五六七八九十百千萬億兆\d]+ |
    [零〇一二兩三四五六七八九十百千萬億兆\d]+分之[零〇一二兩三四五六七八九十百千萬億兆\d]+
"""
_MEASURE_UNIT = r"(?:tCO2e|CO2e|二氧化碳當量|新台幣|新臺幣|美元|NTD|TWD|兆元|億元|千萬元|百萬元|十萬元|萬元|兆|億|萬|仟元|千元|元|公噸|噸|公斤|公克|克|kg|kWh|MWh|GWh|GJ|TJ|焦耳|CMD|公升|立方公尺|立方米|m3|m³|噸水|平方公里|公頃|平方公尺|平方米|㎡|m2|公里|公尺|公分|毫米|米|坪|千瓦|kW|MW|GW|kWp|度|℃|°C|攝氏|ppm|ppb|年|個月|月|日|天|週|周|季|期|階段|張|項|件|人次|人|名|家|處|座|場|批|份|套|筆|戶|輛|台|條|小時|工時|人時|次|%|％)"
_MEASURE_QUANTITY_UNIT_RE = re.compile(
    rf"(?:約|逾|超過|超逾|達|至少|近|約莫|數|累計|合計|共|每年|每月|每季)?\s*{_MEASURE_VALUE}\s*{_MEASURE_UNIT}",
    re.IGNORECASE,
)
_MEASURE_STANDALONE_RE = re.compile(
    rf"""
        %|％ |
        {_MEASURE_RATIO} |
        {_MEASURE_CHINESE_NUM} |
        tCO2e|CO2e|二氧化碳當量 |
        kWh|MWh|GWh|GJ|TJ|CMD |
        ppm|ppb |
        ℃|°C
    """,
    re.IGNORECASE | re.VERBOSE,
)


def has_measurable_signal(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(_MEASURE_STANDALONE_RE.search(text) or _MEASURE_QUANTITY_UNIT_RE.search(text))


# feat_E: commitment verb signal (rule-based) — ported from official notebook Phase B-2
_COMMIT_STRONG_RE = re.compile(r"""
    承諾|宣示|宣告|宣布|宣佈|
    將於|將在|擬於|預計於|預定於|規劃於|計畫於|
    預計|預定|預計將|預定將|規劃將|計畫將|
    致力於|力求
""", re.VERBOSE)
_COMMIT_WEAK_RE = re.compile(r"目標|設定|訂定|設立|制定|建立|規劃|計畫|決議", re.VERBOSE)
_COMMIT_FUTURE_RE = re.compile(r"""
    未來|短期|中期|長期|階段|期程|路徑|藍圖|
    20\d{2}|民國\s*\d+|\d+\s*年(?:前|底|內)?|
    [零〇一二兩三四五六七八九十百千萬億兆]+\s*年(?:前|底|內)?
""", re.VERBOSE)
_COMMIT_ESG_RE = re.compile(r"""
    淨零|零碳|減碳|減量|減排|降碳|碳中和|碳盤查|盤查|查證|確信|
    溫室氣體|範疇一|範疇二|範疇三|SBTi|RE100|再生能源|綠電|節能|節水|
    汰換|導入|取得|使用|提高|提升|降低|減少|增加|改善|完成|達成|實現|
    供應鏈|人權|職安|安全衛生|多元共融|生物多樣性|零毀林|永續|氣候|TCFD|TNFD|ISSB|IFRS|碳費
""", re.VERBOSE)
_COMMIT_FALSE_RE = re.compile(r"""
    生產計畫|排程資訊|管理計畫|研究計畫|計畫書|專案計畫|
    重大議題管理項目|目標設定情形|風險矩陣圖|氣候風險與機會矩陣圖|
    預期損失|預期信用損失|預期成本|預期衝擊|預期風險|預估損失|預估成本
""", re.VERBOSE)


def has_commitment_signal(text: str) -> bool:
    if not isinstance(text, str):
        return False
    strong = bool(_COMMIT_STRONG_RE.search(text))
    false_ctx = bool(_COMMIT_FALSE_RE.search(text))
    if strong and not false_ctx:
        return True
    if not bool(_COMMIT_WEAK_RE.search(text)):
        return False
    has_future = bool(_COMMIT_FUTURE_RE.search(text))
    has_esg = bool(_COMMIT_ESG_RE.search(text))
    has_measure = has_measurable_signal(text)
    if false_ctx and not (has_future and has_esg):
        return False
    return bool(has_future or (has_esg and has_measure))


def build_hybrid_feature_map(csv_path: str) -> dict:
    """Load GPT-4o hybrid feature CSV → {str(id): {feat_A, feat_B, feat_C, feat_D, feat_E}}.

    Caller should pass config.HYBRID_TRAIN_FEAT_CSV for training/val data and
    config.HYBRID_TEST_FEAT_CSV for test data.  For samples not in the CSV
    (e.g. synthetic data) _resolve_hybrid_feats() falls back to rule-based C/E
    and defaults A=B=D=0.
    """
    import csv as _csv
    feat_map: dict = {}
    feat_cols = [
        "feat_A_is_historical", "feat_B_is_risk_disclosure",
        "feat_C_is_measurable", "feat_D_has_hedging_words", "feat_E_has_commitment_verb",
    ]
    try:
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                sid = str(row.get("id", "")).strip()
                if not sid:
                    continue
                feat_map[sid] = {col: int(float(row[col])) for col in feat_cols if col in row}
    except FileNotFoundError:
        print(f"[hybrid] WARNING: feature CSV not found: {csv_path}")
    return feat_map


def _resolve_hybrid_feats(text: str, sample_id: str, hybrid_map: Optional[dict]) -> dict:
    """Return a 5-feature dict for one sample.

    If hybrid_map has an entry for sample_id, use it (all 5 features).
    Otherwise compute feat_C/E rule-based and set A=B=D=0 (safe defaults).
    """
    if hybrid_map and sample_id in hybrid_map:
        return hybrid_map[sample_id]
    feat_c = 1 if has_measurable_signal(text) else 0
    feat_e = 1 if has_commitment_signal(text) else 0
    return {
        "feat_A_is_historical": 0,
        "feat_B_is_risk_disclosure": 0,
        "feat_C_is_measurable": feat_c,
        "feat_D_has_hedging_words": 0,
        "feat_E_has_commitment_verb": feat_e,
    }


def build_hybrid_prefix(feats: dict) -> str:
    """Build a 5-token prefix from the feature dict: '[A=0][B=0][C=1][D=0][E=1] '."""
    a = feats.get("feat_A_is_historical", 0)
    b = feats.get("feat_B_is_risk_disclosure", 0)
    c = feats.get("feat_C_is_measurable", 0)
    d = feats.get("feat_D_has_hedging_words", 0)
    e = feats.get("feat_E_has_commitment_verb", 0)
    return f"[A={a}][B={b}][C={c}][D={d}][E={e}] "


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
