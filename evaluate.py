"""
evaluate.py — Load saved model and run full evaluation on test split.

Usage:
    python evaluate.py --data data/raw/esg4k.json --checkpoint models/best.pt
"""

import argparse
from pathlib import Path

import torch
import numpy as np

import sys, os
sys.path.append(os.path.dirname(__file__))
from configs import config
from models.model import ESGMultiTaskModel
from utils.dataset import get_dataloaders
from utils.metrics import (
    compute_all_metrics,
    print_metrics,
    per_task_classification_report,
)
from train import build_criteria, evaluate as eval_fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate ESG Multi-task Classifier")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=str(config.MODELS_DIR / "best.pt"))
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader = get_dataloaders(Path(args.data), batch_size=args.batch_size)
    print(f"Test samples: {len(test_loader.dataset)}")

    model = ESGMultiTaskModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    criteria = build_criteria()
    test_loss, test_metrics = eval_fn(model, test_loader, device, criteria)

    print(f"\nTest loss: {test_loss:.4f}")
    print_metrics(test_metrics)

    # Detailed per-task classification report
    model.eval()
    all_preds = {t: [] for t in config.TASK_NAMES}
    all_golds = {t: [] for t in config.TASK_NAMES}

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            for i, task in enumerate(config.TASK_NAMES):
                all_preds[task].append(logits[task].argmax(-1).cpu().numpy())
                all_golds[task].append(labels[:, i].numpy())

    all_preds = {t: np.concatenate(v) for t, v in all_preds.items()}
    all_golds = {t: np.concatenate(v) for t, v in all_golds.items()}
    per_task_classification_report(all_preds, all_golds)


if __name__ == "__main__":
    main()
