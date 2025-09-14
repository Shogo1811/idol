# === 6.2: フィルタリング（高額データ除外） ===
import pandas as pd

df = pd.read_csv("../output/comparison.csv")

filtered = df[
    (df["Actual Price"] < 250_000_000) & (df["Predicted Price"] < 250_000_000)
]

print("✅ フィルタ済みデータ:")
print(filtered.head())
