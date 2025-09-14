# === 7.1: 軸設定と散布図（1,000万円単位で表示） ===
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# データ読み込み（単位はそのまま「円」）
df = pd.read_csv("../output/comparison_filtered.csv")

plt.figure(figsize=(8, 8))
plt.scatter(df["Actual Price"], df["Predicted Price"], alpha=0.7)

# 軸の範囲設定（0〜2億円）
plt.xlim(0, 2e8)
plt.ylim(0, 2e8)

# 軸目盛（5000万円間隔）
ticks = np.arange(0, 2.1e8, 5e7)  # 50,000,000単位
plt.xticks(ticks)
plt.yticks(ticks)


# 軸ラベルを1,000万円単位で整形
def format_million(value, _):
    return f"{int(value / 1e6):,}"  # 例: 100000000 → "100"


ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_million))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_million))

# ラベルとタイトル
plt.xlabel("Actual Price (×1,000,000 円)")
plt.ylabel("Predicted Price (×1,000,000 円)")
plt.title("Step 1: Axis Config (in Millions of Yen)")
plt.grid(True)
plt.show()
