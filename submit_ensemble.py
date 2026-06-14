"""
submit_ensemble.py — Task-specialized ensemble submission.

Loads FOUR checkpoints, one expert per subtask (each trained with
`train.py --focus_task <task>`), and builds the submission by taking every
task's prediction from its own specialist:

    promise_status        ← commitment expert  (best_commitment.pt)
    evidence_status       ← evidence   expert  (best_evidence.pt)
    evidence_quality      ← clarity    expert  (best_clarity.pt)
    verification_timeline ← timeline   expert  (best_timeline.pt)

The same field-dependency rules as submit.py are then enforced.

Usage (Colab):
    python submit_ensemble.py \
        --data data/raw/vpesg4k_test_2000.json \
        --commitment_ckpt /content/best_commitment.pt \
        --evidence_ckpt   /content/best_evidence.pt \
        --clarity_ckpt    /content/best_clarity.pt \
        --timeline_ckpt   /content/best_timeline.pt \
        --output /content/submission_ensemble.csv
"""

import argparse
import csv
import gc
from pathlib import Path

import torch

import sys, os
sys.path.append(os.path.dirname(__file__))
from configs import config
from models.model import ESGMultiTaskModel
from utils.dataset import load_raw_samples, normalise_field
from utils.text_clean import build_tokenizer, preprocess_text
# Reuse decode maps + competition rule logic from the single-model script.
from submit import DECODE, _apply_rules


@torch.no_grad()
def infer_task_logits(ckpt_path, task, samples, tokenizer, device, batch_size):
    """Load one expert checkpoint, return its logits for a SINGLE task.

    Only that task's head is used; the model is freed before returning so the
    four experts never sit in memory at once.
    """
    model = ESGMultiTaskModel().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()
    print(f"[{task}] loaded expert: {ckpt_path}")

    chunks = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        texts = [preprocess_text(normalise_field(s.get(config.TEXT_FIELD, "")), s)
                 for s in batch]
        enc = tokenizer(
            texts,
            max_length=config.MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        chunks.append(logits[task].cpu())

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return torch.cat(chunks, dim=0)


def main():
    parser = argparse.ArgumentParser(description="ESG Task-Specialized Ensemble Submission")
    parser.add_argument("--data", type=str, required=True,
                        help="Test JSON file (e.g. vpesg4k_test_2000.json)")
    parser.add_argument("--commitment_ckpt", type=str, required=True)
    parser.add_argument("--evidence_ckpt",   type=str, required=True)
    parser.add_argument("--clarity_ckpt",    type=str, required=True)
    parser.add_argument("--timeline_ckpt",   type=str, required=True)
    parser.add_argument("--output",     type=str, default="submission_ensemble.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = build_tokenizer()  # shared tokenizer (raw when domain tokens off)

    samples = load_raw_samples(Path(args.data), max_samples=100_000)
    print(f"Test samples: {len(samples)}")
    all_ids = [s.get("id", 12001 + i) for i, s in enumerate(samples)]

    # Each task's logits come from its own expert checkpoint.
    ckpt_for = {
        "commitment": args.commitment_ckpt,
        "evidence":   args.evidence_ckpt,
        "clarity":    args.clarity_ckpt,
        "timeline":   args.timeline_ckpt,
    }
    all_logits = {
        task: infer_task_logits(ckpt_for[task], task, samples, tokenizer, device, args.batch_size)
        for task in config.TASK_NAMES
    }

    preds = {t: all_logits[t].argmax(dim=-1).tolist() for t in config.TASK_NAMES}
    preds = _apply_rules(preds, all_logits)

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
    with open(out_path, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"CSV lines (header + data): {len(lines)}")
    print("First 3 rows:")
    for ln in lines[1:4]:
        print(" ", ln.strip())


if __name__ == "__main__":
    main()
