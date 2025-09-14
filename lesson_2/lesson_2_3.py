from datetime import datetime

import pandas as pd

# 前ステップと同じ読み込み（または `lesson_2_2.py` の出力を読み込んでもOK）
df = pd.read_csv("../input/Tokyo_20242_20242.csv", encoding="cp932")
df.columns = df.columns.str.replace("\ufeff", "")
columns_to_use = [
    "取引価格（総額）",
    "面積（㎡）",
    "最寄駅：距離（分）",
    "建築年",
    "市区町村名",
]
df = df[columns_to_use].dropna()

# 整数型へ変換
df["建築年"] = df["建築年"].str.replace("年", "", regex=False)
df["建築年"] = pd.to_numeric(df["建築年"], errors="coerce")
df = df.dropna(subset=["建築年"])
df["建築年"] = df["建築年"].astype(int)

df["築年数"] = datetime.now().year - df["建築年"]

# === カテゴリ変数（市区町村名）を数値に変換 ===
# ① ラベルエンコーディング（文字列 → 数値）
df["市区町村コード"] = df["市区町村名"].astype("category").cat.codes

# ② ワンホットエンコーディングしたい場合（列が増えるので注意）
# df = pd.get_dummies(df, columns=["市区町村名"])

print("▼ 市区町村名 → 数値変換後")
print(df[["市区町村名", "市区町村コード"]].drop_duplicates().head())
