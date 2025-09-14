from datetime import datetime

import pandas as pd

# lesson_2_3.pyの処理後のデータを想定（再実装してもOK）
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

df["建築年"] = df["建築年"].str.replace("年", "", regex=False)
df["建築年"] = pd.to_numeric(df["建築年"], errors="coerce")
df = df.dropna(subset=["建築年"])
df["建築年"] = df["建築年"].astype(int)

df["築年数"] = datetime.now().year - df["建築年"]

df["市区町村コード"] = df["市区町村名"].astype("category").cat.codes

# 不要な文字列列は削除（例：市区町村名）
df = df.drop(columns=["市区町村名"])
# 築年数を使用するため、削除する
df = df.drop(columns=["建築年"])

# データを出力する
output_path = "../output/processed_data.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"▼ 前処理済みデータを保存しました: {output_path}")
