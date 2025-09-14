# === 6.4: フィルタ後データを保存 ===
import pandas as pd

df = pd.read_csv("../output/comparison.csv")
filtered = df[
    (df["Actual Price"] < 250_000_000) & (df["Predicted Price"] < 250_000_000)
]

filtered.to_csv("../output/comparison_filtered.csv", index=False, encoding="utf-8-sig")
print("✅ comparison_filtered.csv を保存しました")
