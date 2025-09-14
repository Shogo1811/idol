# === 7.3: 理想線と仕上げ ===
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

df = pd.read_csv("../output/comparison_filtered.csv")

plt.figure(figsize=(8, 8))
ax = plt.gca()

# 散布図
plt.scatter(
    df["Actual Price"],
    df["Predicted Price"],
    color="orange",
    alpha=0.7,
    label="Predicted vs Actual",
)

# 軸設定
x_min, x_max = 0, 2e8
y_min, y_max = 0, 2e8
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ticks = np.arange(0, 2.1e8, 5e7)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

# ラベル整形
formatter = ticker.FuncFormatter(lambda val, _: f"{int(val / 1e6):,}")
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)

# グリッドと理想線
ax.grid(which="major", linestyle="--", linewidth=0.5)
plt.plot([x_min, x_max], [y_min, y_max], "--r", linewidth=2, label="Ideal Prediction")

# ラベル
plt.xlabel("Actual Price (x 1,000,000)")
plt.ylabel("Predicted Price (x 1,000,000)")
plt.title("Final Visualization: Actual vs Predicted")
plt.legend()
plt.show()
