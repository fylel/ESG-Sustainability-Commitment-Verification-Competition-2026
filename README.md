# ESG Sustainability Commitment Verification Competition 2026

Multi-task BERT classifier for verifying ESG (Environmental, Social, Governance) sustainability commitments, built for the VeriPromiseESG4K competition dataset.

## Task Overview

Given an ESG commitment statement, the model simultaneously predicts four dimensions:

| Task | Field | Classes |
|------|-------|---------|
| `commitment` | promise_status | Yes / No |
| `evidence` | evidence_status | Yes / No |
| `clarity` | evidence_quality | Clear / Not Clear / Misleading |
| `timeline` | verification_timeline | already / within_2_years / between_2_and_5_years / more_than_5_years |

## Model Architecture

Shared BERT encoder ([`hfl/chinese-macbert-base`](https://huggingface.co/hfl/chinese-macbert-base)) with one independent classification head per task.

```
[CLS] text [SEP]
      │
 BERT encoder (shared)
      │
 [CLS] hidden (768-d)
      │
 ┌────┴────┬──────────┬──────────┐
 head_0   head_1    head_2     head_3
commitment evidence  clarity   timeline
  (2)       (2)       (3)       (4)
```

## Project Structure

```
├── configs/
│   └── config.py          # All hyperparameters and label mappings
├── data/
│   └── raw/               # Place your dataset here
├── models/
│   └── model.py           # ESGMultiTaskModel definition
├── utils/
│   ├── dataset.py         # DataLoader and tokenization
│   ├── metrics.py         # Evaluation metrics
│   └── tokenizer.py       # Tokenizer utilities
├── train.py               # Training loop with Optuna support
├── evaluate.py            # Detailed evaluation with competition scoring
├── predict.py             # Inference script
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --data data/raw/vpesg_4k_train_1000.json --epochs 23
```

### Hyperparameter Tuning (Optuna)

```bash
python train.py --data data/raw/vpesg_4k_train_1000.json --tune --n_trials 20 --tune_epochs 5
```

### Prediction

```bash
python predict.py --data data/raw/vpesg_4k_train_1000.json --model_path models/best.pt
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Pretrained model | `hfl/chinese-macbert-base` |
| Max sequence length | 256 |
| Batch size | 16 |
| Learning rate | 1e-5 |
| Epochs | 23 |
| Early stopping patience | 5 |

## Evaluation

The competition uses a weighted F1 score across all four tasks:

| Task | Weight |
|------|--------|
| clarity | 0.35 |
| evidence | 0.30 |
| commitment | 0.20 |
| timeline | 0.15 |
