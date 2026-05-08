"""
Evaluation metrics for multi-task ESG classification.
"""

from typing import Dict, List
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config


def compute_task_metrics(
    preds: np.ndarray,
    golds: np.ndarray,
    task: str,
) -> Dict[str, float]:
    """
    Compute accuracy & macro-F1 for a single task, ignoring samples
    whose gold label equals IGNORE_INDEX (-1).
    """
    mask = golds != config.IGNORE_INDEX
    if mask.sum() == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0, "support": 0}

    p, g = preds[mask], golds[mask]
    return {
        "accuracy": float(accuracy_score(g, p)),
        "f1_macro": float(f1_score(g, p, average="macro", zero_division=0)),
        "support": int(mask.sum()),
    }


def compute_all_metrics(
    all_preds: Dict[str, np.ndarray],
    all_golds: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Return metrics dict keyed by task name."""
    results: Dict[str, Dict[str, float]] = {}
    for task in config.TASK_NAMES:
        results[task] = compute_task_metrics(
            all_preds[task], all_golds[task], task
        )
    return results


def print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print per-task metrics."""
    print("\n" + "=" * 55)
    print(f"{'Task':<14} {'Acc':>8} {'F1-macro':>10} {'Support':>9}")
    print("-" * 55)
    for task in config.TASK_NAMES:
        m = metrics[task]
        print(f"{task:<14} {m['accuracy']:>8.4f} {m['f1_macro']:>10.4f} {m['support']:>9d}")
    print("=" * 55 + "\n")


def per_task_classification_report(
    all_preds: Dict[str, np.ndarray],
    all_golds: Dict[str, np.ndarray],
) -> None:
    """Print sklearn classification_report per task."""
    inv_maps = {}
    for task, mapping in config.LABEL_MAPS.items():
        inv = {v: k for k, v in mapping.items() if v != config.IGNORE_INDEX}
        inv_maps[task] = inv

    for task in config.TASK_NAMES:
        mask = all_golds[task] != config.IGNORE_INDEX
        if mask.sum() == 0:
            print(f"\n[{task}] No valid samples.\n")
            continue
        p, g = all_preds[task][mask], all_golds[task][mask]
        inv = inv_maps[task]
        labels = sorted(inv.keys())
        target_names = [inv[l] for l in labels]
        print(f"\n{'─' * 10}  {task}  {'─' * 10}")
        print(classification_report(g, p, labels=labels,
                                    target_names=target_names,
                                    zero_division=0))
