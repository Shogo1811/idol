# === 6.3: 散布図を表示（簡易版） ===
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

filtered = pd.read_csv("../output/comparison.csv")
filtered = filtered[
    (filtered["Actual Price"] < 250_000_000)
    & (filtered["Predicted Price"] < 250_000_000)
]

plt.figure(figsize=(8, 8))
plt.scatter(
    filtered["Actual Price"],
    filtered["Predicted Price"],
    alpha=0.7,
    label="Predicted vs Actual",
)


def format_million_yen(value, _):
    return f"{int(value / 1e6):,}"


ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_million_yen))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_million_yen))

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Scatter Plot (Filtered)")
plt.legend()
plt.grid(True)
plt.show()
