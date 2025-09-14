# === 3.1: データ読み込み ===
# === 3.1: 前処理済みCSVから X, y を保存 ===
import joblib
import pandas as pd

# 前処理済みデータを読み込む
df = pd.read_csv("../output/processed_data.csv", encoding="utf-8-sig")

# ターゲット（取引価格）と特徴量に分ける
y = df["取引価格（総額）"]
X = df.drop("取引価格（総額）", axis=1)

# pkl形式で保存
joblib.dump(X, "../output/X.pkl")
joblib.dump(y, "../output/y.pkl")

print("▼ Xとyをpkl形式で保存しました")
print("Xの形状:", X.shape)
print("yの形状:", y.shape)


# 保存された前処理済みのX, yを読み込み
X = joblib.load("../output/X.pkl")
y = joblib.load("../output/y.pkl")

# 確認
print("Xの形状:", X.shape)
print("yの形状:", y.shape)
