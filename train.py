"""
train.py — Multi-task training loop for VeriPromiseESG4K.

Usage (Colab):
    !python train.py --data data/raw/esg4k.json --epochs 10

Hyperparameter tuning (Optuna):
    !python train.py --data data/raw/esg4k.json --tune --n_trials 20 --tune_epochs 5

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
from sklearn.utils.class_weight import compute_class_weight
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import sys, os
sys.path.append(os.path.dirname(__file__))
from configs import config
from models.model import ESGMultiTaskModel
from utils.dataset import get_dataloaders
from utils.metrics import compute_all_metrics, print_metrics
from evaluate import evaluate_detailed


# ──────────────────────────────────────────────────────────────────────
# Loss helper
# ──────────────────────────────────────────────────────────────────────

def build_criteria(train_ds, device) -> dict:
    """One CrossEntropyLoss per task with class weights computed from training data."""
    criteria = {}
    for task in config.TASK_NAMES:
        task_labels = [train_ds.labels[i][task] for i in range(len(train_ds))]
        valid = [l for l in task_labels if l != config.IGNORE_INDEX]
        num_classes = config.NUM_CLASSES[task]
        weights = np.ones(num_classes, dtype=np.float64)
        present = np.unique(valid)
        if len(present) > 0:
            present_weights = compute_class_weight("balanced", classes=present, y=valid)
            for cls, w in zip(present, present_weights):
                weights[cls] = w
        if task == "clarity":
            weights[config.CLARITY_MAP["Misleading"]] = 1e-4  # near-zero to avoid NaN when full batch is Misleading
        weight_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        criteria[task] = nn.CrossEntropyLoss(
            weight=weight_tensor, ignore_index=config.IGNORE_INDEX
        )
    if config.USE_SPAN_AUX:
        criteria["__span__"] = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    if config.USE_KEYWORD_AUX:
        criteria["__keyword__"] = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    if config.USE_TEMPORAL_AUX:
        criteria["__temporal__"] = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    return criteria


def combined_loss(
    logits: dict,
    labels: torch.Tensor,
    criteria: dict,
    span_labels: torch.Tensor = None,
    keyword_labels: torch.Tensor = None,
    temporal_labels: torch.Tensor = None,
) -> torch.Tensor:
    """
    logits         : {task_name: (B, C), optionally span/keyword logits}
    labels         : (B, num_tasks)  columns in TASK_NAMES order
    span_labels    : (B, 4) [promise_start, promise_end, evidence_start, evidence_end]
    keyword_labels : (B, seq_len)  1=keyword, 0=non-keyword, -1=ignore
    """
    total = torch.tensor(0.0, device=labels.device)
    for i, task in enumerate(config.TASK_NAMES):
        task_labels = labels[:, i]
        task_loss = criteria[task](logits[task], task_labels)
        total = total + config.TASK_LOSS_WEIGHTS[task] * task_loss

    if (config.USE_SPAN_AUX and span_labels is not None
            and "__span__" in criteria):
        span_ce = criteria["__span__"]
        span_terms = [
            (logits["promise_start"],  span_labels[:, 0]),
            (logits["promise_end"],    span_labels[:, 1]),
            (logits["evidence_start"], span_labels[:, 2]),
            (logits["evidence_end"],   span_labels[:, 3]),
        ]
        span_loss_sum = torch.tensor(0.0, device=labels.device)
        valid_count = 0
        for logit, lbl in span_terms:
            if (lbl != -1).any():
                span_loss_sum = span_loss_sum + span_ce(logit, lbl)
                valid_count += 1
        if valid_count > 0:
            total = total + config.SPAN_LOSS_WEIGHT * span_loss_sum / valid_count

    if (config.USE_KEYWORD_AUX and keyword_labels is not None
            and "__keyword__" in criteria):
        kw_logits = logits["keyword"]           # (B, seq_len, 2)
        B, S, C = kw_logits.shape
        kw_loss = criteria["__keyword__"](
            kw_logits.view(B * S, C), keyword_labels.view(B * S)
        )
        total = total + config.KEYWORD_LOSS_WEIGHT * kw_loss

    if (config.USE_TEMPORAL_AUX and temporal_labels is not None
            and "__temporal__" in criteria):
        tm_logits = logits["temporal"]          # (B, seq_len, 3)
        B, S, C = tm_logits.shape
        tm_loss = criteria["__temporal__"](
            tm_logits.view(B * S, C), temporal_labels.view(B * S)
        )
        total = total + config.TEMPORAL_LOSS_WEIGHT * tm_loss

    return total


# ──────────────────────────────────────────────────────────────────────
# Optuna objective
# ──────────────────────────────────────────────────────────────────────

def objective(trial, args, device, train_loader, val_loader, train_ds):
    encoder_lr   = trial.suggest_float("encoder_lr",   1e-5, 5e-5, log=True)
    head_lr      = trial.suggest_float("head_lr",      1e-4, 1e-3, log=True)
    dropout      = trial.suggest_float("dropout",      0.1,  0.4)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)

    model    = ESGMultiTaskModel(dropout=dropout).to(device)
    criteria = build_criteria(train_ds, device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.heads.parameters(),   "lr": head_lr},
    ], weight_decay=weight_decay)

    total_steps  = len(train_loader) * args.tune_epochs
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[encoder_lr, head_lr],
        total_steps=total_steps,
        pct_start=max(warmup_steps / total_steps, 1e-4),
        anneal_strategy="cos",
    )

    best_score = 0.0
    for epoch in range(1, args.tune_epochs + 1):
        train_one_epoch(model, train_loader, device, optimizer, criteria)
        _, _, val_preds, val_golds = evaluate(model, val_loader, device, criteria)
        val_score = evaluate_detailed(val_preds, val_golds)["final_weighted_score"]
        scheduler.step()

        best_score = max(best_score, val_score)
        trial.report(val_score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_score


# ──────────────────────────────────────────────────────────────────────
# Train / Validate one epoch
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, device, optimizer, criteria):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for input_ids, attention_mask, labels, span_labels, keyword_labels, temporal_labels in tqdm(loader, desc="Train"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        span_labels = span_labels.to(device)
        keyword_labels = keyword_labels.to(device)
        temporal_labels = temporal_labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = combined_loss(logits, labels, criteria, span_labels, keyword_labels, temporal_labels)
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

    for input_ids, attention_mask, labels, span_labels, keyword_labels, temporal_labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        span_labels = span_labels.to(device)
        keyword_labels = keyword_labels.to(device)
        temporal_labels = temporal_labels.to(device)

        logits = model(input_ids, attention_mask)
        loss = combined_loss(logits, labels, criteria, span_labels, keyword_labels, temporal_labels)

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
    return avg_loss, metrics, all_preds, all_golds


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ESG Multi-task Classifier")
    parser.add_argument("--data", type=str, required=True, help="Path to JSON/JSONL data")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--save_path", type=str, default=str(config.MODELS_DIR / "best.pt"))
    parser.add_argument("--augment", nargs="+", default=None, metavar="FILE",
                        help="Augmented JSON files appended to training data")
    parser.add_argument("--val_data", type=str, default=None, metavar="FILE",
                        help="External validation JSON (uses all train data for training, no split)")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--tune_epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument("--resume", type=str, default=None, metavar="CKPT",
                        help="Resume training from checkpoint (loads weights, uses low LR cosine decay)")
    parser.add_argument("--merge_val", action="store_true",
                        help="Merge val_data into training pool (final submission: train on all labeled data)")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model  : {config.PRETRAINED_MODEL}")
    print(f"LR     : {args.lr}  |  Batch: {args.batch_size}  |  Epochs: {args.epochs}")
    print(f"Dropout: {config.CLASSIFIER_DROPOUT}  |  WeightDecay: {config.WEIGHT_DECAY}  |  MaxLen: {config.MAX_SEQ_LEN}")

    # Data
    train_loader, val_loader, test_loader, train_ds = get_dataloaders(
        Path(args.data), batch_size=args.batch_size, return_train_ds=True,
        augment_paths=args.augment,
        val_path=Path(args.val_data) if args.val_data else None,
        merge_val=args.merge_val,
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")

    # ── Optuna tuning mode ────────────────────────────────────────────
    if args.tune:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _trial_callback(study, trial):
            print(f"[Trial {trial.number:>2}] score={trial.value:.5f} | "
                  + " | ".join(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in trial.params.items()))

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=config.SEED),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        )
        study.optimize(
            lambda trial: objective(trial, args, device, train_loader, val_loader, train_ds),
            n_trials=args.n_trials,
            show_progress_bar=True,
            callbacks=[_trial_callback],
        )
        print("\n=== Best Trial ===")
        best = study.best_trial
        print(f"  Score : {best.value:.5f}")
        print(f"  Params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
        return

    # Model
    model = ESGMultiTaskModel().to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"[resume] randomly initialised: {missing}")
        if unexpected:
            print(f"[resume] ignored (not in model): {unexpected}")
        print(f"Resumed from {args.resume}")
    criteria = build_criteria(train_ds, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler: cosine decay when resuming (no warmup), OneCycleLR for fresh training
    total_steps = len(train_loader) * args.epochs
    if args.resume:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-7
        )
    else:
        warmup_steps = int(total_steps * config.WARMUP_RATIO)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            pct_start=warmup_steps / total_steps, anneal_strategy="cos",
        )

    writer = SummaryWriter(log_dir=str(config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S")))
    writer.add_text("config", f"model={config.PRETRAINED_MODEL}, batch={config.BATCH_SIZE}, lr={config.LEARNING_RATE}, epochs={args.epochs}")

    # Training loop
    best_val_score = 0.0
    save_path = Path("/content/best.pt")
    patience = config.EARLY_STOPPING_PATIENCE
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 15}  Epoch {epoch}/{args.epochs}  {'=' * 15}")

        train_loss = train_one_epoch(model, train_loader, device, optimizer, criteria)
        val_loss, val_metrics, val_preds, val_golds = evaluate(model, val_loader, device, criteria)
        val_results = evaluate_detailed(val_preds, val_golds)
        val_score = val_results["final_weighted_score"]

        scheduler.step()

        # Log
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("score/val_weighted", val_score, epoch)
        for task in config.TASK_NAMES:
            writer.add_scalar(f"acc/{task}", val_metrics[task]["accuracy"], epoch)
            writer.add_scalar(f"f1/{task}", val_metrics[task]["f1_macro"], epoch)

        print(f"Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}  |  Val score: {val_score:.5f}")
        print_metrics(val_metrics)

        # Save best / early stopping (based on val score = competition metric)
        if val_score > best_val_score:
            best_val_score = val_score
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"✓ Model saved → {save_path}  (score={val_score:.5f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()

    # Final test evaluation
    print("\n" + "=" * 20 + "  Test Set  " + "=" * 20)
    if len(test_loader.dataset) == 0:
        print("No test set (TEST_RATIO=0), skipping.")
    else:
        if not save_path.exists():
            print("Warning: no checkpoint found, evaluating with current weights")
        else:
            model.load_state_dict(torch.load(save_path, map_location=device))
        test_loss, test_metrics, test_preds, test_golds = evaluate(model, test_loader, device, criteria)
        test_results = evaluate_detailed(test_preds, test_golds)
        print(f"Test loss: {test_loss:.4f}  |  Test score: {test_results['final_weighted_score']:.5f}")
        print_metrics(test_metrics)


if __name__ == "__main__":
    main()
