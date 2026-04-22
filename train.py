"""
train.py — Multi-task training loop for VeriPromiseESG4K.

Usage (Colab):
    !python train.py --data data/raw/esg4k.json --epochs 10

Usage (script):
    python train.py --data data/raw/esg4k.json
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import sys, os
sys.path.append(os.path.dirname(__file__))
from configs import config
from models.model import ESGMultiTaskModel
from utils.dataset import get_dataloaders
from utils.metrics import compute_all_metrics, print_metrics


# ──────────────────────────────────────────────────────────────────────
# Loss helper
# ──────────────────────────────────────────────────────────────────────

def build_criteria() -> dict:
    """One CrossEntropyLoss per task, each ignoring IGNORE_INDEX."""
    return {
        task: nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)
        for task in config.TASK_NAMES
    }


def combined_loss(logits: dict, labels: torch.Tensor, criteria: dict) -> torch.Tensor:
    """
    logits : {task_name: (B, C)}
    labels : (B, num_tasks)  columns in TASK_NAMES order
    """
    total = torch.tensor(0.0, device=labels.device)
    for i, task in enumerate(config.TASK_NAMES):
        task_labels = labels[:, i]
        task_loss = criteria[task](logits[task], task_labels)
        total = total + config.TASK_LOSS_WEIGHTS[task] * task_loss
    return total


# ──────────────────────────────────────────────────────────────────────
# Train / Validate one epoch
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, device, optimizer, criteria):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for input_ids, attention_mask, labels in tqdm(loader, desc="Train"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = combined_loss(logits, labels, criteria)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * input_ids.size(0)
        n_samples += input_ids.size(0)

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device, criteria):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    all_preds = {t: [] for t in config.TASK_NAMES}
    all_golds = {t: [] for t in config.TASK_NAMES}

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask)
        loss = combined_loss(logits, labels, criteria)

        running_loss += loss.item() * input_ids.size(0)
        n_samples += input_ids.size(0)

        for i, task in enumerate(config.TASK_NAMES):
            preds = logits[task].argmax(dim=-1).cpu().numpy()
            golds = labels[:, i].cpu().numpy()
            all_preds[task].append(preds)
            all_golds[task].append(golds)

    # concatenate
    all_preds = {t: np.concatenate(v) for t, v in all_preds.items()}
    all_golds = {t: np.concatenate(v) for t, v in all_golds.items()}

    avg_loss = running_loss / max(n_samples, 1)
    metrics = compute_all_metrics(all_preds, all_golds)
    return avg_loss, metrics


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ESG Multi-task Classifier")
    parser.add_argument("--data", type=str, required=True, help="Path to JSON/JSONL data")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        Path(args.data), batch_size=args.batch_size
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")

    # Model
    model = ESGMultiTaskModel().to(device)
    criteria = build_criteria()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_steps / total_steps, anneal_strategy="cos",
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=str(config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S")))

    # Training loop
    best_val_loss = float("inf")
    save_path = config.MODELS_DIR / "best.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 15}  Epoch {epoch}/{args.epochs}  {'=' * 15}")

        train_loss = train_one_epoch(model, train_loader, device, optimizer, criteria)
        val_loss, val_metrics = evaluate(model, val_loader, device, criteria)

        scheduler.step()

        # Log
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        for task in config.TASK_NAMES:
            writer.add_scalar(f"acc/{task}", val_metrics[task]["accuracy"], epoch)
            writer.add_scalar(f"f1/{task}", val_metrics[task]["f1_macro"], epoch)

        print(f"Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")
        print_metrics(val_metrics)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✓ Model saved → {save_path}")

    writer.close()

    # Final test evaluation
    print("\n" + "=" * 20 + "  Test Set  " + "=" * 20)
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_metrics = evaluate(model, test_loader, device, criteria)
    print(f"Test loss: {test_loss:.4f}")
    print_metrics(test_metrics)


if __name__ == "__main__":
    main()
