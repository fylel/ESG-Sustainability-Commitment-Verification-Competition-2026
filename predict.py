"""
predict.py — Run inference on a single text or batch of texts.

Usage:
    python predict.py --checkpoint models/best.pt --text "公司承諾2025年前達成碳中和目標"
"""

import argparse
import torch
from transformers import BertTokenizer

import sys, os
sys.path.append(os.path.dirname(__file__))
from configs import config
from models.model import ESGMultiTaskModel


def build_inverse_maps():
    """Build int → label name for each task (excluding IGNORE_INDEX)."""
    inv = {}
    for task, mapping in config.LABEL_MAPS.items():
        inv[task] = {v: k for k, v in mapping.items() if v != config.IGNORE_INDEX}
    return inv


def predict(text: str, model, tokenizer, device, inv_maps):
    """Return a dict {task: predicted_label_name}."""
    enc = tokenizer(
        text,
        max_length=config.MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    results = {}
    for task in config.TASK_NAMES:
        pred_id = logits[task].argmax(dim=-1).item()
        results[task] = inv_maps[task].get(pred_id, f"UNKNOWN({pred_id})")
    return results


def main():
    parser = argparse.ArgumentParser(description="ESG Predict")
    parser.add_argument("--checkpoint", type=str, default=str(config.MODELS_DIR / "best.pt"))
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL)
    model = ESGMultiTaskModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    inv_maps = build_inverse_maps()
    results = predict(args.text, model, tokenizer, device, inv_maps)

    print(f"\nInput: {args.text}")
    print("-" * 40)
    for task, label in results.items():
        print(f"  {task:<14}: {label}")


if __name__ == "__main__":
    main()
