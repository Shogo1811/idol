# === 7.2: 軸ラベルの整形 ===
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

df = pd.read_csv("../output/comparison_filtered.csv")

plt.figure(figsize=(8, 8))
ax = plt.gca()

plt.scatter(df["Actual Price"], df["Predicted Price"], alpha=0.7)

# 軸範囲
ax.set_xlim(0, 2e8)
ax.set_ylim(0, 2e8)

# 目盛り
ticks = np.arange(0, 2.1e8, 5e7)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

# 整形：1000万単位
formatter = ticker.FuncFormatter(lambda val, _: f"{int(val / 1e6):,}")
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)

plt.xlabel("Actual Price (x 1,000,000)")
plt.ylabel("Predicted Price (x 1,000,000)")
plt.title("Step 2: Axis Label Formatting")
plt.grid(True)
plt.show()
