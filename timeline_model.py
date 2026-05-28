# ============================================================
# ESG Verification Timeline Classification
# TF-IDF + Time Feature Engineering + Class Weight + Rule Fix
# ============================================================

import json
import re
import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# 1. 讀取資料
# ============================================================

file_path = "data/raw/vpesg_4k_train_1000.json"  # 修正：原為 /mnt/data/ 路徑

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("原始資料筆數：", len(df))
print("\npromise_status 分布：")
print(df["promise_status"].value_counts(dropna=False))
print("\nverification_timeline 分布：")
print(df["verification_timeline"].value_counts(dropna=False))


# ============================================================
# 2. 只取 promise_status = Yes 的資料訓練 timeline
#    promise_status = No 的 timeline 皆為空值，排除以免干擾
# ============================================================

df_t = df[df["promise_status"] == "Yes"].copy()
df_t = df_t[
    df_t["verification_timeline"].notna() &
    (df_t["verification_timeline"].astype(str).str.strip() != "")
].copy()

print("\n用於 timeline 訓練的資料筆數：", len(df_t))
print("\nTimeline 類別分布：")
print(df_t["verification_timeline"].value_counts())


# ============================================================
# 3. 建立 timeline_input
#
#    修正：移除 promise_string 與 evidence_string。
#    這兩欄是人工標注產物，不在 USED_FIELDS 中，
#    測試資料不提供，使用會造成訓練/推論分布不一致。
#    改為只用原始 data 欄位。
# ============================================================

df_t["timeline_input"] = df_t["data"].fillna("").astype(str)


# ============================================================
# 4. 時間特徵工程
# ============================================================

CURRENT_YEAR = datetime.date.today().year  # 修正：原為硬碼 2024


def extract_time_features(text):
    text = str(text)

    years = [int(y) for y in re.findall(r"20[2-5]\d", text)]

    max_year = max(years) if years else 0
    min_year = min(years) if years else 0

    if years:
        max_year_gap = max_year - CURRENT_YEAR
        min_year_gap = min_year - CURRENT_YEAR
    else:
        max_year_gap = 0
        min_year_gap = 0

    return {
        "has_year":       int(len(years) > 0),
        "year_count":     len(years),
        "max_year":       max_year,
        "min_year":       min_year,
        "max_year_gap":   max_year_gap,
        "min_year_gap":   min_year_gap,

        "has_2024":        int(2024 in years),
        "has_2025_2026":   int(any(y in [2025, 2026] for y in years)),
        "has_2027_2029":   int(any(2027 <= y <= 2029 for y in years)),
        "has_2030_plus":   int(any(y >= 2030 for y in years)),
        "has_2050":        int(2050 in years),

        "has_already_words": int(any(w in text for w in [
            "已", "已完成", "已達成", "完成", "達成", "截至",
            "目前", "現行", "2024年", "2024 年", "實績"
        ])),
        "has_future_words": int(any(w in text for w in [
            "預計", "目標", "承諾", "將", "持續", "未來",
            "規劃", "推動", "致力", "期望", "預定"
        ])),
        "has_short_words": int(any(w in text for w in [
            "短期", "一年內", "兩年內", "2年內", "2025", "2026"
        ])),
        "has_mid_words": int(any(w in text for w in [
            "中期", "中程", "2027", "2028", "2029"
        ])),
        "has_long_words": int(any(w in text for w in [
            "長期", "長程", "2030", "2035", "2040", "2050",
            "淨零", "碳中和", "永續發展", "長期目標"
        ])),

        "has_percentage": int(bool(re.search(r"\d+(\.\d+)?\s*%", text))),
        "has_number":     int(bool(re.search(r"\d+", text))),
    }


time_features = df_t["timeline_input"].apply(extract_time_features).apply(pd.Series)
df_t = pd.concat([df_t.reset_index(drop=True), time_features.reset_index(drop=True)], axis=1)
feature_cols = list(time_features.columns)

print("\n新增時間特徵欄位：", feature_cols)


# ============================================================
# 5. 切分訓練集與測試集
# ============================================================

X = df_t[["timeline_input"] + feature_cols]
y = df_t["verification_timeline"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n訓練集類別分布：")
print(y_train.value_counts())
print("\n測試集類別分布：")
print(y_test.value_counts())


# ============================================================
# 6. 建立模型：TF-IDF + 數值時間特徵 + Logistic Regression
# ============================================================

preprocess = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                max_features=3000,         # 修正：8000→3000，~650筆訓練資料避免過擬合
                ngram_range=(1, 2),
                token_pattern=r"(?u)\b\w+\b"
            ),
            "timeline_input"
        ),
        (
            "num",
            StandardScaler(),
            feature_cols
        )
    ]
)

model = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(
        C=0.5,              # 修正：加入，within_2_years 只有 13 筆，加強正則化防噪音
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        # 修正：移除 multi_class="auto"（deprecated，lbfgs 預設即處理多分類）
    ))
])


# ============================================================
# 7. 訓練模型
# ============================================================

model.fit(X_train, y_train)
pred_ml = model.predict(X_test)

print("\n==============================")
print("純機器學習模型結果")
print("==============================")
print("Accuracy:", accuracy_score(y_test, pred_ml))
print("\nClassification Report:")
print(classification_report(y_test, pred_ml))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_ml))


# ============================================================
# 8. Rule-based 後處理修正
#
#    修正：調整優先順序，先判斷 already，再判斷年份。
#    原版先判斷 2030+，導致「已達成 2030 年目標」被錯判為 more_than_5_years。
# ============================================================

def rule_fix_timeline(text, pred):
    text = str(text)
    years = [int(y) for y in re.findall(r"20[2-5]\d", text)]

    # ① 先判斷 already：明確完成的陳述不受年份干擾
    if any(w in text for w in [
        "已完成", "已達成", "已通過", "已取得",
        "截至2024", "截至 2024", "2024 年已", "2024年已",
        "目前已", "實績", "完成"
    ]):
        return "already"

    # ② 長期目標（2030+）
    if any(y >= 2030 for y in years) or any(w in text for w in [
        "2050", "2040", "2035", "2030", "淨零", "碳中和", "長期目標", "長期"
    ]):
        return "more_than_5_years"

    # ③ 短期目標（2025-2026）
    if any(y in [2025, 2026] for y in years) or any(w in text for w in [
        "一年內", "兩年內", "2年內", "短期目標", "短期"
    ]):
        return "within_2_years"

    # ④ 中期目標（2027-2029）
    if any(2027 <= y <= 2029 for y in years) or any(w in text for w in [
        "中期", "中程"
    ]):
        return "between_2_and_5_years"

    return pred


pred_rule_fixed = [
    rule_fix_timeline(text, pred)
    for text, pred in zip(X_test["timeline_input"], pred_ml)
]

print("\n==============================")
print("機器學習 + Rule-based 修正後結果")
print("==============================")
print("Accuracy:", accuracy_score(y_test, pred_rule_fixed))
print("\nClassification Report:")
print(classification_report(y_test, pred_rule_fixed))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_rule_fixed))


# ============================================================
# 9. 整理測試集預測結果
# ============================================================

result_df = X_test.copy()
result_df["true_timeline"] = y_test.values
result_df["pred_ml"] = pred_ml
result_df["pred_rule_fixed"] = pred_rule_fixed

display_cols = ["timeline_input", "true_timeline", "pred_ml", "pred_rule_fixed"]

print("\n預測結果前 10 筆：")
print(result_df[display_cols].head(10))


# ============================================================
# 10. 找出預測錯誤的資料
# ============================================================

wrong_df = result_df[result_df["true_timeline"] != result_df["pred_rule_fixed"]].copy()
print("\nRule-based 修正後仍預測錯誤的筆數：", len(wrong_df))
print("\n錯誤樣本前 20 筆：")
print(wrong_df[display_cols].head(20))


# ============================================================
# 11. 輸出結果 CSV
# ============================================================

result_df.to_csv("timeline_prediction_result.csv", index=False, encoding="utf-8-sig")
wrong_df.to_csv("timeline_wrong_cases.csv", index=False, encoding="utf-8-sig")

print("\n已輸出：timeline_prediction_result.csv / timeline_wrong_cases.csv")
