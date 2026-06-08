# ESG 競賽專案現況

## 競賽說明

- 任務：多標籤分類，給定一段 ESG 報告文字（`data` 欄位），預測 4 個子任務
- 訓練資料有 `promise_string` / `evidence_string`（標注者從 `data` 摘錄的句子），**測試資料沒有**
- 評分公式：`0.20×commitment + 0.15×timeline + 0.30×evidence + 0.35×clarity`

### 4 個子任務

| 任務 | 欄位 | 類別數 | 類別列表 | 評分權重 |
|------|------|--------|---------|---------|
| commitment | promise_status | 2 | Yes / No | 0.20 |
| timeline | verification_timeline | 5 | already / within_2 / between_2_5 / more_than_5 / **N/A** | 0.15 |
| evidence | evidence_status | 3 | Yes / No / **N/A** | 0.30 |
| clarity | evidence_quality | 4 | Clear / Not Clear / Misleading / **N/A** | 0.35 |

> **N/A 是競賽真實評分類別**（官方 EVAL_FIELDS 明確列出），以 Macro F1 等權計算。若模型從不預測 N/A，對應類別 F1=0，直接拉低分數。

---

## 模型架構

```
raw data → BERT (hfl/chinese-macbert-base)
                │
         [CLS] hidden ──→ commitment head (2)  Yes / No
                      ──→ evidence head   (3)  Yes / No / N/A
                      ──→ clarity head    (4)  Clear / Not Clear / Misleading / N/A
                      ──→ timeline head   (5)  already / within_2 / between_2_5 / more_than_5 / N/A
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
- 增強資料（Google Drive symlink，放於 `esg_data/`）：

| 檔名 | 筆數 | 增強目標 | 內容說明 |
|------|------|---------|---------|
| `aug_timeline_within2.json` | 25 | timeline = within_2_years | 舊版 102 筆同質模板替換為 25 筆多樣版；12 行業、多元句型、Clear/Not Clear/Misleading 均有 |
| `aug_timeline_between_clear.json` | 80 | timeline = between_2_5, Clear | 2027–2029 年承諾，有具體數字與明確執行進度 |
| `aug_timeline_between_mixed.json` | 80 | timeline = between_2_5, Not Clear/Misleading | 2027–2029 年承諾；Not Clear 26 筆 + Misleading 18 筆，補邊界樣本 |
| `aug_timeline_morethan.json` | 80 | timeline = more_than_5_years | 2030 年後長期目標 |
| `aug_commitment_no.json` | 100 | promise_status = No | 沒有做出承諾的 ESG 段落；原始資料 186 筆偏少 |
| `aug_evidence_no.json` | 100 | evidence_status = No | 有承諾但無佐證；timeline 四類均勻分布（各 25 筆）|
| `aug_quality_misleading.json` | 100 | evidence_quality = Misleading | 表面像承諾但實際誤導（舊版 timeline 全是 already 已修正）；timeline 四類各 25 筆 |
| `aug_quality_notclear.json` | 100 | evidence_quality = Not Clear | 承諾模糊無量化目標；timeline 分布對齊真實資料（between_2_5=45, already=28, more_than_5=27）|

增強資料由 AI 生成，merge 進原始資料後一起切分 train/val/test。

---

## 成績歷史

| 版本 | commitment | evidence | clarity | timeline | Total |
|------|-----------|---------|---------|---------|-------|
| Baseline | 0.64 | 0.72 | 0.73 | 0.38 | ~0.68 |
| + span aux | 0.837 | 0.865 | 0.746 | 0.478 | 0.759 |
| + val_score early stopping | 0.880 | 0.849 | 0.744 | 0.533 | 0.771 |
| + resume 接續訓練 | **0.884** | **0.906** | **0.782** | **0.794** | **0.842** |
| + keyword aux（退步，已停用） | 0.813 | 0.851 | 0.757 | 0.301 | 0.728 |
| + 新增強資料全套（8 個 aug 檔）| 0.780 | 0.891 | **0.834** | 0.483 | 0.787 |
| + within2 換新版 25 筆多樣版 | 0.821 | 0.847 | **0.871** | 0.466 | 0.793 |

### 最新結果（within2 換新版 25 筆，Total = 0.793）

```mermaid
xychart-beta
    title "Latest Model F1 by Task (Total = 0.793)"
    x-axis ["commitment (w=0.20)", "evidence (w=0.30)", "clarity (w=0.35)", "timeline (w=0.15)"]
    y-axis "Macro F1" 0 --> 1
    bar [0.821, 0.847, 0.871, 0.466]
    line [0.793, 0.793, 0.793, 0.793]
```

```mermaid
xychart-beta
    title "Clarity Sub-class F1"
    x-axis ["Clear", "Not Clear", "Misleading"]
    y-axis "F1" 0 --> 1
    bar [0.900, 0.730, 0.970]
```

---

## Colab 執行流程

```python
# 1. 掛載 Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone
import os; os.chdir('/content')
!rm -rf /content/translation-transformer
!git clone https://github.com/fylel/ESG-Sustainability-Commitment-Verification-Competition-2026.git /content/translation-transformer

# 3. 建 symlink（主資料 + 增強資料 + 驗證資料）
import os
dst_main = '/content/translation-transformer/data/raw/vpesg_4k_train_1000.json'
if not os.path.lexists(dst_main):
    os.symlink('/content/drive/MyDrive/esg_data/vpesg_4k_train_1000.json', dst_main)
dst_val = '/content/translation-transformer/data/raw/vpesg4k_val_1000.json'
if not os.path.lexists(dst_val):
    os.symlink('/content/drive/MyDrive/esg_data/vpesg4k_val_1000.json', dst_val)
for f in [
    'aug_timeline_within2.json',
    'aug_timeline_between_clear.json',
    'aug_timeline_between_mixed.json',
    'aug_timeline_morethan.json',
    'aug_commitment_no.json',
    'aug_evidence_no.json',
    'aug_quality_misleading.json',
    'aug_quality_notclear.json',
]:
    dst = f'/content/translation-transformer/data/raw/{f}'
    if not os.path.lexists(dst):
        os.symlink(f'/content/drive/MyDrive/esg_data/{f}', dst)

# 4. 安裝套件
!pip install transformers torch scikit-learn tensorboard tqdm matplotlib optuna -q

# 5a. 全新訓練（第一次 or 資料有大幅變動）
%cd /content/translation-transformer
!python train.py --data data/raw/vpesg_4k_train_1000.json \
  --val_data data/raw/vpesg4k_val_1000.json \
  --augment data/raw/aug_timeline_within2.json \
            data/raw/aug_timeline_between_clear.json \
            data/raw/aug_timeline_between_mixed.json \
            data/raw/aug_timeline_morethan.json \
            data/raw/aug_commitment_no.json \
            data/raw/aug_evidence_no.json \
            data/raw/aug_quality_misleading.json \
            data/raw/aug_quality_notclear.json \
  --epochs 30

# 5b. 繼續訓練（從現有 checkpoint 接續，用較低 LR）
# 先把上次存的 .pt 複製到 /content/best.pt，再執行：
%cd /content/translation-transformer
!python train.py --data data/raw/vpesg_4k_train_1000.json \
  --val_data data/raw/vpesg4k_val_1000.json \
  --augment data/raw/aug_timeline_within2.json \
            data/raw/aug_timeline_between_clear.json \
            data/raw/aug_timeline_between_mixed.json \
            data/raw/aug_timeline_morethan.json \
            data/raw/aug_commitment_no.json \
            data/raw/aug_evidence_no.json \
            data/raw/aug_quality_misleading.json \
            data/raw/aug_quality_notclear.json \
  --resume /content/best.pt \
  --lr 5e-6 \
  --epochs 15

# 6. 評估 + 圖表
%matplotlib inline
%cd /content/translation-transformer
!python evaluate.py --data data/raw/vpesg_4k_train_1000.json \
  --val_data data/raw/vpesg4k_val_1000.json \
  --checkpoint /content/best.pt \
  --augment data/raw/aug_timeline_within2.json \
            data/raw/aug_timeline_between_clear.json \
            data/raw/aug_timeline_between_mixed.json \
            data/raw/aug_timeline_morethan.json \
            data/raw/aug_commitment_no.json \
            data/raw/aug_evidence_no.json \
            data/raw/aug_quality_misleading.json \
            data/raw/aug_quality_notclear.json
from IPython.display import Image
Image('/content/translation-transformer/f1_scores.png')

# 7. 存模型 + 圖表
from datetime import datetime; import shutil
ts = datetime.now().strftime("%m%d_%H%M")
shutil.copy('/content/best.pt',
            f'/content/drive/MyDrive/esg_data/best_{ts}.pt')
shutil.copy('/content/translation-transformer/f1_scores.png',
            f'/content/drive/MyDrive/esg_data/f1_scores_{ts}.png')
```

---

## 資料分布：增強前 vs 增強後

總樣本數：原始 1,000 筆 → 增強後 **1,742 筆**（+742）

### verification_timeline

| 類別 | 原始 | 原始 % | 增強後 | 增強後 % | 變化 |
|------|-----:|-------:|-------:|---------:|------|
| already | 366 | 36.6% | 445 | 25.5% | ↓（相對比例下降，絕對數增加）|
| between_2_and_5_years | 238 | 23.8% | 492 | 28.2% | ↑ |
| more_than_5_years | 197 | 19.7% | 353 | 20.3% | ≈ |
| within_2_years | **13** | **1.3%** | **166** | **9.5%** | ↑↑ 大幅補足 |
| 空值/N/A | 186 | 18.6% | 286 | 16.4% | ≈ |

### evidence_quality
 
| 類別 | 原始 | 原始 % | 增強後 | 增強後 % | 變化 |
|------|-----:|-------:|-------:|---------:|------|
| Clear | 552 | 55.2% | 743 | 42.7% | ↓（稀釋）|
| Not Clear | 124 | 12.4% | 261 | 15.0% | ↑ |
| Misleading | **1** | **0.1%** | **119** | **6.8%** | ↑↑ 大幅補足 |
| 空值/N/A | 323 | 32.3% | 619 | 35.5% | ≈ |

### promise_status / evidence_status

| 欄位 | 類別 | 原始 | 增強後 |
|------|------|-----:|-------:|
| promise_status | No | 186 (18.6%) | 286 (16.4%) |
| promise_status | Yes | 814 (81.4%) | 1456 (83.6%) |
| evidence_status | No | 137 (13.7%) | 333 (19.1%) ↑ |
| evidence_status | Yes | 677 (67.7%) | 1123 (64.5%) |

---

## 當前弱點分析（基於 0.842 成績）

| 弱點 | 影響程度 | 說明 | 處理狀態 |
|------|---------|------|---------|
| Clarity「Not Clear」F1=0.45 | 最高（權重 0.35） | 和 Clear 邊界模糊 | ✅ 已補 aug_quality_notclear.json（100 筆）|
| Timeline「between_2_and_5_years」F1=0.63 | 中 | recall=0.55 | ✅ 已補 aug_timeline_between_mixed.json（80 筆 Not Clear/Misleading）|
| Misleading 舊資料 timeline 全是 already | 高 | 假關聯 | ✅ 已替換 aug_quality_misleading.json（timeline 均勻）|
| within_2_years F1=1.00（虛假） | 中 | 競賽真實資料可能崩 | 待觀察 |
| Commitment「No」F1=0.81 | 低 | 類別不平衡 | 已有 aug_commitment_no.json |

### ⚠ 增強資料的兩個根本陷阱

**1. Misleading F1=0.97 是虛假高分 → 決策：放棄訓練 Misleading**
- 原始資料只有 1 筆 Misleading；增強後 119 筆中 ~118 筆來自 GPT 模板
- test set 的 Misleading 樣本幾乎全是同款 GPT 寫法，模型學到的是「這種模板 = Misleading」
- 競賽真實資料不會長這樣，實際泛化能力未知
- 競賽確認使用 Macro F1（`average="macro", zero_division=0`）：每個類別等權重，少樣本不會自動縮小比重
- Misleading 在 clarity 內佔 1/3，換算總分約 11.7%（0.35 × 1/3）
- 陷阱：若競賽 test 有 Misleading 樣本卻預測錯，F1=0，直接拉低分數；原始只有 1 筆無法可靠訓練，故放棄
- **✅ 決策：保留 aug_quality_misleading.json**；Macro F1 下少樣本≠小影響，放棄等同穩定扣 11.7% 總分

**2. within_2 補足為什麼反而退步（違反直覺）**
- 13 筆原始 → 153 筆（92% 是 AI 模板），樣本多樣性極低
- GPT 傾向固定句式（「預計於 2025/2026 年…」），140 筆差異極小
- 模型 overfit 模板，把有相近寫法的 between_2_5 也誤判為 within_2 → precision=0.38
- 根本原因：**within_2 年份訊號本來就清晰**（2025/2026），13 筆真實樣本已夠，大量同質模板反而干擾
- 相比之下，Not Clear / Misleading 語義邊界本來就模糊，更多例子確實幫助學邊界
- **結論：增強資料有效的前提是多樣性，不是數量**

## 架構設計決策

### N/A 處理：為何不用階層式模型（commitment 結果往下傳）

**問法**：既然 commitment=No → evidence/clarity/timeline 全部 N/A 是邏輯規則，為什麼不把 commitment 頭的輸出傳給下游頭？

**問題一：訓練初期不穩定**
- 訓練早期 commitment 頭預測幾乎全錯
- 若把錯誤預測傳給 evidence/clarity/timeline 頭，三個頭同時學到錯誤信號，全部學壞
- 錯誤從 commitment 層開始向下 cascade，難以收斂

**問題二：梯度無法流通**
- 傳遞方式若用 hard argmax（取最高分類別）→ 不可微，梯度無法回傳給 commitment 頭
- 傳遞 soft logits/probability 雖可微，但早期 noise 仍大

**解法：獨立頭 + 推論後處理**
- 4 個頭各自獨立讀同一個 `[CLS]`，互不影響，訓練穩定
- 推論完成後套確定性規則（post-processing）：
  - `commitment_pred = "No"` → 強制 evidence / clarity / timeline = `"N/A"`
  - `evidence_pred = "No"` → 強制 clarity = `"N/A"`
- 規則 100% 正確，不依賴模型學到這個 pattern

**為何還要訓練 N/A 類別（不用 IGNORE_INDEX）**
- 4 個頭彼此獨立，evidence 頭不知道 commitment 頭預測了什麼
- 若 commitment 預測錯（判 Yes 但真實 No），post-processing 不會觸發，evidence 頭就拿「從沒訓練過 N/A」的結果去輸出
- 讓模型訓練 N/A 類別 = 讓 evidence/clarity/timeline 頭也學到文字特徵 → N/A 的關聯，作為 fallback
- 同時，`encode_labels` 的 dependency rules 把原始資料的 `""` 補成 N/A index，確保訓練資料有 N/A 樣本

---

## 已知問題 / 待處理

- keyword aux 實作完成但會造成退步（timeline 從 0.533 跌到 0.301），原因待查，目前停用（USE_KEYWORD_AUX = False）

### 下次訓練方向（最優先）

**問題：timeline 整體偏弱（Macro F1=0.466）**
- within_2 precision=0.31，recall=0.85 → 仍過度預測
- 原因：換新版 within2（25 筆）後仍有問題；`aug_quality_misleading.json` 和 `aug_evidence_no.json` 各有 25 筆 within_2 → 增強總量仍達 75 筆
- between_2_5 F1=0.46、more_than_5 F1=0.38 同樣偏弱

**待處理改善（優先序）：**
1. 改 `dataset.py` → val/test 只用原始資料，取得誠實評估基準
2. 降低其他 aug 檔中 within_2 的樣本比例（misleading/evidence_no 各有 25 筆）
3. 提高 timeline loss weight：0.10 → 0.15（對齊競賽評分）

**保留的成果：**
- clarity 增強有效：Not Clear 0.45 → 0.65，Misleading 0.62 → 0.97
- 其餘 7 個 aug 檔保留

**下次訓練指令（Step 5a，全 8 個 aug 檔，within2 已換新版 25 筆多樣版）：**
```
!python train.py --data data/raw/vpesg_4k_train_1000.json \
  --augment data/raw/aug_timeline_within2.json \
            data/raw/aug_timeline_between_clear.json \
            data/raw/aug_timeline_between_mixed.json \
            data/raw/aug_timeline_morethan.json \
            data/raw/aug_commitment_no.json \
            data/raw/aug_evidence_no.json \
            data/raw/aug_quality_misleading.json \
            data/raw/aug_quality_notclear.json \
  --epochs 30
```
> aug_timeline_within2.json → 舊版 102 筆同質模板已替換為新版 25 筆多樣版（12 行業、多元句型）
> aug_quality_misleading.json → 保留；Macro F1 下 Misleading 佔 11.7% 總分，不能放棄
