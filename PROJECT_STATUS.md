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

## 成績歷史

| 版本 | commitment | evidence | clarity | timeline | Total |
|------|-----------|---------|---------|---------|-------|
| Baseline | 0.64 | 0.72 | 0.73 | 0.38 | ~0.68 |
| + span aux | 0.837 | 0.865 | 0.746 | 0.478 | 0.759 |
| + val_score early stopping | 0.880 | 0.849 | 0.744 | 0.533 | 0.771 |
| + resume 接續訓練（現在最佳） | **0.884** | **0.906** | **0.782** | **0.794** | **0.842** |
| + keyword aux（退步，已停用） | 0.813 | 0.851 | 0.757 | 0.301 | 0.728 |

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
dst_main = '/content/translation-transformer/data/raw/vpesg_4k_train_1000.json'
if not os.path.lexists(dst_main):
    os.symlink('/content/drive/MyDrive/esg_data/vpesg_4k_train_1000.json', dst_main)
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

## 當前弱點分析（基於 0.842 成績）

| 弱點 | 影響程度 | 說明 |
|------|---------|------|
| Clarity「Not Clear」F1=0.45 | 最高（權重 0.35） | 和 Clear 邊界模糊，樣本只有 22 筆 |
| Timeline「between_2_and_5_years」F1=0.63 | 中 | recall=0.55，常被誤判為 more_than_5_years |
| within_2_years F1=1.00（虛假） | 中 | 幾乎全是增強資料，真實資料可能崩掉 |
| Commitment「No」F1=0.81 | 低 | 42 vs 190 類別不平衡 |

## 已知問題 / 待處理

- keyword aux 實作完成但會造成退步（timeline 從 0.533 跌到 0.301），原因待查，目前停用（USE_KEYWORD_AUX = False）
- 下次方向選項：
  1. 針對「Not Clear」補增強資料（最優先，權重最高）
  2. 針對「between_2_and_5_years」補增強資料或調整 loss 權重
  3. 調低 KEYWORD_LOSS_WEIGHT（0.10 → 0.05）重試 keyword aux
  4. 若 timeline 仍卡關，考慮 RAG + GPT 處理 timeline 子任務
