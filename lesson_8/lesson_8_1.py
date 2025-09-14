import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# === 1. CSVファイルを読み込む ===
file_path = "../input/Tokyo_20242_20242.csv"  # CSVファイルのパスを指定
# MEMO 文字コードがutf-8の場合
# df = pd.read_csv(file_path, sep="\t", encoding="utf-8")  # タブ区切り
# MEMO 文字コードがcp932の場合
df = pd.read_csv(file_path, encoding="cp932")  # sepは指定しない（デフォルトで","）
df.columns = df.columns.str.replace("\ufeff", "")  # 念のためBOM除去

# 必要な列を選択
columns_to_use = [
    "取引価格（総額）",
    "面積（㎡）",
    "最寄駅：距離（分）",
    "建築年",
    "市区町村名",
]
df = df[columns_to_use]

print(df)


# === 2. データの前処理 ===
# "建築年"を数値化
df["建築年"] = pd.to_numeric(df["建築年"].str.replace("年", ""), errors="coerce")


# "最寄駅：距離（分）"を数値化
def parse_distance(distance):
    if isinstance(distance, str):
        if "H" in distance:  # "1H" や "1H30分"
            parts = distance.replace("分", "").split("H")
            hours = int(parts[0]) * 60
            minutes = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            return hours + minutes
        elif "～" in distance:  # "30分～60分"
            parts = distance.replace("分", "").split("～")
            return (int(parts[0]) + int(parts[1])) / 2
        elif "分" in distance:  # "5分"
            return int(distance.replace("分", ""))
    return distance  # 数値の場合はそのまま返す


df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].apply(parse_distance)

# 欠損値を削除
df = df.dropna()

# 市区町村名をOne-Hot Encoding
df = pd.get_dummies(df, columns=["市区町村名"], drop_first=True)

# 特徴量（X）とターゲット（y）に分割
X = df.drop("取引価格（総額）", axis=1)
y = df["取引価格（総額）"]

# === 3. データ分割 ===
# ランダムサンプリング
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# データを分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 分割後のデータサイズを確認
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# === 4. モデルの構築とトレーニング ===
# ランダムフォレスト
# モデルの構築
model = RandomForestRegressor(random_state=42)
# トレーニングデータで学習
model.fit(X_train, y_train)
# モデルの概要を表示
print("Model Summary:", model)

# === 5. テストデータで予測 ===
y_pred = model.predict(X_test)
y_pred = np.round(y_pred, 2)

# === 6. 結果を評価 ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")


# === 7. 実際の価格と予測価格を比較 ===
comparison = pd.DataFrame(
    {
        "Actual Price": y_test.values,  # y_test.values で値を取得
        "Predicted Price": y_pred,
    }
)

# Predicted Price を整数に揃える
comparison["Predicted Price"] = comparison["Predicted Price"].astype(int)

# print(comparison.head())

# # === 8. 実際の価格 vs 予測価格をプロット ===
# # 散布図を描画
plt.figure(figsize=(8, 8))
plt.scatter(
    comparison["Actual Price"],
    comparison["Predicted Price"],
    alpha=0.7,
    label="Predicted vs Actual",
)


print(comparison["Actual Price"].head())
print(comparison["Predicted Price"].head())

filtered_comparison = comparison[
    (comparison["Actual Price"] < 250000000)
    & (comparison["Predicted Price"] < 250000000)
]

# グラフ作成
plt.scatter(
    filtered_comparison["Actual Price"],
    filtered_comparison["Predicted Price"],
    alpha=0.7,
)
plt.xscale("log")
plt.yscale("log")

# グラフの描画
plt.figure(figsize=(8, 8))

# データの最大値を確認して範囲を設定
x_min, x_max = 0, 2e8  # 0 から 200,000,000
y_min, y_max = 0, 2e8
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 目盛りを一定間隔で設定
ticks = np.arange(0, 2.1e8, 5e7)  # 50,000,000単位で目盛りを設定
ax = plt.gca()
ax.set_xticks(ticks)
ax.set_yticks(ticks)


# ラベルをカスタマイズ（例: 10,000,000 -> 1000）
def format_ticks(value, _):
    return f"{int(value / 1e6):,}"  # 1,000,000 で割り整数表示


ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

# グリッドを設定
ax.grid(which="major", linestyle="--", linewidth=0.5)

# 散布図を描画
plt.scatter(
    filtered_comparison["Actual Price"],
    filtered_comparison["Predicted Price"],
    color="orange",
    alpha=0.7,
    label="Predicted vs Actual",
)

# 理想的な予測ラインを描画
plt.plot([x_min, x_max], [y_min, y_max], "--r", linewidth=2, label="Ideal Prediction")

# 軸ラベルとタイトル
plt.xlabel("Actual Price (x 1,000,000)")
plt.ylabel("Predicted Price (x 1,000,000)")
plt.title("Comparison of Actual vs Predicted Prices")
plt.legend()
plt.show()
