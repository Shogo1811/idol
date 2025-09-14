import pandas as pd

file_path = "lesson_1.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig")

# 特定の列だけ抽出
print(df["取引価格（総額）"])

# 条件フィルタリング（例：価格が5000万円以上）
filtered = df[df["取引価格（総額）"] >= 50000000]
print(filtered)

# 複数条件（かつ・または）
filtered2 = df[(df["取引価格（総額）"] >= 50000000) & (df["最寄駅：距離（分）"] <= 10)]
print(filtered2)
