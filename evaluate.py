"""
evaluate.py — Load saved model and run full evaluation on test split.

Includes:
  - Per-task classification report
  - Competition weighted score (Macro F1 based)
  - F1 Score visualization chart

Usage:
    python evaluate.py --data data/raw/vpesg_4k_train_1000.json --checkpoint models/best.pt
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

import sys, os
sys.path.append(os.path.dirname(__file__))
from configs import config
from models.model import ESGMultiTaskModel
from utils.dataset import get_dataloaders


# ──────────────────────────────────────────────────────────────────────
# Detailed evaluation with Macro/Micro F1 + competition score
# ──────────────────────────────────────────────────────────────────────

def evaluate_detailed(all_preds, all_golds):
    """Compute per-task Macro F1, Micro F1, classification report, and final weighted score."""
    results = {}

    inv_maps = {}
    for task, mapping in config.LABEL_MAPS.items():
        inv_maps[task] = {v: k for k, v in mapping.items() if v != config.IGNORE_INDEX}

    for task in config.TASK_NAMES:
        mask = all_golds[task] != config.IGNORE_INDEX
        if mask.sum() == 0:
            results[task] = {
                "macro_f1": 0.0, "micro_f1": 0.0,
                "report": "No valid samples.", "weight": config.EVAL_WEIGHTS[task],
            }
            continue

        p, g = all_preds[task][mask], all_golds[task][mask]
        inv = inv_maps[task]
        labels = sorted(inv.keys())
        target_names = [inv[l] for l in labels]

        macro_f1 = float(f1_score(g, p, average="macro", zero_division=0))
        micro_f1 = float(f1_score(g, p, average="micro", zero_division=0))
        report = classification_report(g, p, labels=labels,
                                       target_names=target_names,
                                       zero_division=0)

        results[task] = {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "report": report,
            "weight": config.EVAL_WEIGHTS[task],
        }

    # Final weighted score (based on Macro F1)
    final_score = sum(
        results[task]["macro_f1"] * config.EVAL_WEIGHTS[task]
        for task in config.TASK_NAMES
    )
    results["final_weighted_score"] = final_score

    return results


# ──────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────

def plot_f1_scores(results, save_path="f1_scores.png"):
    """Draw F1 Score by Task bar chart."""
    display_names = {
        "commitment": "promise_status",
        "timeline": "verification_timeline",
        "evidence": "evidence_status",
        "clarity": "evidence_quality",
    }

    fields = config.TASK_NAMES
    macro_f1s = [results[f]["macro_f1"] for f in fields]
    micro_f1s = [results[f]["micro_f1"] for f in fields]
    weights = [config.EVAL_WEIGHTS[f] for f in fields]

    x = range(len(fields))
    width = 0.3

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width / 2 for i in x], macro_f1s, width,
                   label="Macro F1", color="steelblue", alpha=0.8)
    bars2 = ax.bar([i + width / 2 for i in x], micro_f1s, width,
                   label="Micro F1", color="coral", alpha=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    final_score = results["final_weighted_score"]
    ax.set_xlabel("Task label")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"F1 Score by Task\nFinal Weighted Score: {final_score:.5f}",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{display_names[f]}\n(w={w})" for f, w in zip(fields, weights)],
                       fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(final_score, color="green", linestyle="--", linewidth=2,
               label=f"Weighted Score={final_score:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"圖表已存到 {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

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

    # Collect predictions
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

    # Detailed evaluation
    results = evaluate_detailed(all_preds, all_golds)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"最終評估結果")
    print(f"{'=' * 60}")

    display_names = {
        "commitment": "promise_status",
        "timeline": "verification_timeline",
        "evidence": "evidence_status",
        "clarity": "evidence_quality",
    }

    for task in config.TASK_NAMES:
        r = results[task]
        print(f"\n--- {display_names[task]} (權重: {r['weight']}) ---")
        print(r["report"])
        print(f"  Macro F1: {r['macro_f1']:.4f}")
        print(f"  Micro F1: {r['micro_f1']:.4f}")

    print(f"\n{'=' * 60}")
    print(f"最終加權分數: {results['final_weighted_score']:.5f}")
    print(f"{'=' * 60}")

    # Plot
    plot_f1_scores(results)


if __name__ == "__main__":
    main()
