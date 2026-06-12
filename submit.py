"""
submit.py — Generate competition submission CSV from test data.

Loads a trained checkpoint, runs batch inference on the test JSON,
applies competition logic rules, and writes a UTF-8 CSV.

Usage (Colab):
    python submit.py \
        --data data/raw/vpesg4k_test_2000.json \
        --checkpoint /content/best.pt \
        --output /content/submission.csv
"""

import argparse
import csv
from pathlib import Path

import torch

import sys, os
sys.path.append(os.path.dirname(__file__))
from configs import config
from models.model import ESGMultiTaskModel
from utils.dataset import load_raw_samples, normalise_field
from utils.text_clean import build_tokenizer, preprocess_text


# ── Decode maps (int → competition label string) ─────────────────────────
DECODE = {
    "commitment": {0: "No",  1: "Yes"},
    "evidence":   {0: "No",  1: "Yes",  2: "N/A"},
    "clarity":    {0: "Clear", 1: "Not Clear", 2: "Misleading", 3: "N/A"},
    "timeline":   {
        0: "already",
        1: "within_2_years",
        2: "between_2_and_5_years",
        3: "more_than_5_years",
        4: "N/A",
    },
}

# N/A indices for each task
_NA = {
    "timeline": config.TIMELINE_MAP["N/A"],
    "evidence":  config.EVIDENCE_MAP["N/A"],
    "clarity":   config.CLARITY_MAP["N/A"],
}


def _best_not_na(logit_row: torch.Tensor, na_idx: int) -> int:
    """argmax excluding the N/A class."""
    masked = logit_row.clone()
    masked[na_idx] = float("-inf")
    return int(masked.argmax())


def _apply_rules(preds: dict, logits: dict) -> dict:
    """Enforce competition field-dependency rules using logits as fallback."""
    yes_commit = config.COMMITMENT_MAP["Yes"]
    no_commit  = config.COMMITMENT_MAP["No"]
    yes_evid   = config.EVIDENCE_MAP["Yes"]

    for i in range(len(preds["commitment"])):
        if preds["commitment"][i] == no_commit:
            # 非承諾段落: all dependent fields → N/A
            preds["timeline"][i] = _NA["timeline"]
            preds["evidence"][i] = _NA["evidence"]
            preds["clarity"][i]  = _NA["clarity"]
        else:
            # promise_status = Yes: timeline must not be N/A
            if preds["timeline"][i] == _NA["timeline"]:
                preds["timeline"][i] = _best_not_na(logits["timeline"][i], _NA["timeline"])

            # evidence_status must be Yes or No (N/A only when commitment=No)
            if preds["evidence"][i] == _NA["evidence"]:
                preds["evidence"][i] = _best_not_na(logits["evidence"][i], _NA["evidence"])

            # evidence=No → clarity must be N/A
            if preds["evidence"][i] != yes_evid:
                preds["clarity"][i] = _NA["clarity"]
            else:
                # evidence=Yes: clarity must not be N/A
                if preds["clarity"][i] == _NA["clarity"]:
                    preds["clarity"][i] = _best_not_na(logits["clarity"][i], _NA["clarity"])

    return preds


def main():
    parser = argparse.ArgumentParser(description="ESG Competition Submission Generator")
    parser.add_argument("--data",       type=str, required=True,
                        help="Test JSON file (e.g. vpesg4k_test_2000.json)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained model checkpoint (.pt)")
    parser.add_argument("--output",     type=str, default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = build_tokenizer()  # shared tokenizer (+ ESG domain tokens)
    model = ESGMultiTaskModel().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device), strict=False
    )
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    samples = load_raw_samples(Path(args.data), max_samples=100_000)
    print(f"Test samples: {len(samples)}")

    all_logits = {t: [] for t in config.TASK_NAMES}
    all_ids: list = []

    for i in range(0, len(samples), args.batch_size):
        batch = samples[i : i + args.batch_size]
        # Same cleaning + company masking as training (per-sample for masking).
        texts = [preprocess_text(normalise_field(s.get(config.TEXT_FIELD, "")), s)
                 for s in batch]
        for j, s in enumerate(batch):
            all_ids.append(s.get("id", 12001 + i + j))

        enc = tokenizer(
            texts,
            max_length=config.MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(enc["input_ids"].to(device),
                           enc["attention_mask"].to(device))
        for task in config.TASK_NAMES:
            all_logits[task].append(logits[task].cpu())

        if (i // args.batch_size + 1) % 10 == 0:
            print(f"  processed {min(i + args.batch_size, len(samples))}/{len(samples)}")

    # Concatenate and decode
    all_logits = {t: torch.cat(v, dim=0) for t, v in all_logits.items()}
    preds = {t: all_logits[t].argmax(dim=-1).tolist() for t in config.TASK_NAMES}

    preds = _apply_rules(preds, all_logits)

    # Write CSV
    fieldnames = [
        "id", "promise_status", "verification_timeline",
        "evidence_status", "evidence_quality",
    ]
    out_path = Path(args.output)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(samples)):
            writer.writerow({
                "id":                    all_ids[i],
                "promise_status":        DECODE["commitment"][preds["commitment"][i]],
                "verification_timeline": DECODE["timeline"][preds["timeline"][i]],
                "evidence_status":       DECODE["evidence"][preds["evidence"][i]],
                "evidence_quality":      DECODE["clarity"][preds["clarity"][i]],
            })

    print(f"\nSaved {len(samples)} rows → {out_path}")

    # Quick sanity check
    with open(out_path, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"CSV lines (header + data): {len(lines)}")
    print("First 3 rows:")
    for ln in lines[1:4]:
        print(" ", ln.strip())


if __name__ == "__main__":
    main()
