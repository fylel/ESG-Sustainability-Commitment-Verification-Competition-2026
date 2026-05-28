# ESG 競賽專案現況

## 競賽說明

- 任務：多標籤分類，給定一段 ESG 報告文字（`data` 欄位），預測 4 個子任務
- 訓練資料有 `promise_string` / `evidence_string`（標注者從 `data` 摘錄的句子），**測試資料沒有**
- 評分公式：`0.20×commitment + 0.15×timeline + 0.30×evidence + 0.35×clarity`

### 4 個子任務

| 任務 | 欄位 | 類別數 | 評分權重 |
|------|------|--------|---------|
| commitment | promise_status | 2 | 0.20 |
| timeline | verification_timeline | 4 | 0.15 |
| evidence | evidence_status | 2 | 0.30 |
| clarity | evidence_quality | 3 | 0.35 |

---

## 模型架構

```
raw data → BERT (hfl/chinese-macbert-base)
                │
         [CLS] hidden ──→ commitment head (2)
                      ──→ evidence head   (2)
                      ──→ clarity head    (3)
                      ──→ timeline head   (4)
                │
         每個 token ──→ promise_span_head  (start/end)  ← 訓練用，推論丟棄
                    ──→ evidence_span_head (start/end)  ← 訓練用，推論丟棄
                    ──→ keyword_head (binary 0/1)       ← 訓練用，推論丟棄
```

### 三層輔助任務邏輯

| 層級 | 任務 | 監督訊號來源 | 解決的問題 |
|------|------|------------|-----------|
| 句子段落級 | Span Extraction Aux | promise_string/evidence_string 的 token 位置 | 模型學會「關鍵句在哪裡」 |
| 詞語級 | Keyword Aux (filtered) | promise/evidence string 內有語意的詞 | 模型學會「哪些詞重要」 |

**filtered 的意思**：停用詞（的、了、在、於…）和標點不會被標為 keyword，保留有語意的詞和數字（如 2030）。

---

## 損失函數

```
Total Loss = 0.20×commitment + 0.35×evidence + 0.35×clarity + 0.10×timeline
           + 0.15×span_aux
           + 0.10×keyword_aux
```

> 注意：timeline 在損失裡是 0.10，但比賽評分是 0.15，這是刻意的設計選擇（之前討論過，先維持現狀）

---

## 訓練設定

- Pretrained: `hfl/chinese-macbert-base`
- MAX_SEQ_LEN: 256
- BATCH_SIZE: 16
- LEARNING_RATE: 1e-5
- EPOCHS: 23（Colab 跑）
- Early stopping: patience=5，以 **val_score（Macro F1）** 為準（不是 val_loss）
- Scheduler: OneCycleLR

---

## 資料

- 主資料：`data/raw/vpesg_4k_train_1000.json`（1000 筆）
- 增強資料（Google Drive symlink）：
  - `augmented_timeline.json`
  - `augmented_commitment.json`
  - `augmented_misleading.json`
  - `augmented_evidence_no.json`
  - `augmented_between.json`
  - `augmented_morethan.json`

增強資料由 AI 生成，merge 進原始資料後一起切分 train/val/test。

---

## 最新成績（加入 span aux 後）

| Task | F1-macro |
|------|---------|
| commitment | 0.837 |
| evidence | 0.865 |
| clarity | 0.746 |
| timeline | 0.478 |
| **Total** | **0.759** |

> Baseline（span aux 前）：commitment=0.64, evidence=0.72, clarity=0.73, timeline=0.38, total≈0.68

---

## Colab 執行流程

```python
# 1. 掛載 Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone
!rm -rf /content/translation-transformer
!git clone https://github.com/fylel/ESG-Sustainability-Commitment-Verification-Competition-2026.git /content/translation-transformer

# 3. 建 symlink（主資料 + 增強資料）
import os
os.symlink('/content/drive/MyDrive/esg_data/vpesg_4k_train_1000.json',
           '/content/translation-transformer/data/raw/vpesg_4k_train_1000.json')
for f in ['augmented_timeline.json','augmented_commitment.json','augmented_misleading.json',
          'augmented_evidence_no.json','augmented_between.json','augmented_morethan.json']:
    os.symlink(f'/content/drive/MyDrive/esg_data/{f}',
               f'/content/translation-transformer/data/raw/{f}')

# 4. 安裝套件
!pip install transformers torch scikit-learn tensorboard tqdm matplotlib optuna -q

# 5. 訓練
%cd /content/translation-transformer
!python train.py --data data/raw/vpesg_4k_train_1000.json \
  --augment data/raw/augmented_timeline.json data/raw/augmented_commitment.json \
            data/raw/augmented_misleading.json data/raw/augmented_evidence_no.json \
            data/raw/augmented_between.json data/raw/augmented_morethan.json \
  --epochs 23

# 6. 評估
!python evaluate.py --data data/raw/vpesg_4k_train_1000.json \
  --checkpoint /content/best.pt \
  --augment data/raw/augmented_timeline.json data/raw/augmented_commitment.json \
            data/raw/augmented_misleading.json data/raw/augmented_evidence_no.json \
            data/raw/augmented_between.json data/raw/augmented_morethan.json

# 7. 存模型
from datetime import datetime; import shutil
shutil.copy('/content/best.pt',
            f'/content/drive/MyDrive/esg_data/best_{datetime.now().strftime("%m%d_%H%M")}.pt')
```

---

## 已知問題 / 待觀察

- `within_2_years` 類別只有 13 筆，timeline F1 仍受限
- keyword aux 是本次新增，尚未跑過完整訓練，效果待觀察
- 若下次要做的事：跑完訓練後比較 keyword aux 加入前後的 val score
