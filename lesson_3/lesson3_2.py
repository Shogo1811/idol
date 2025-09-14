# === 3.2: データ分割 ===
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_distance(value):
    if isinstance(value, str):
        value = value.strip()
        if "H" in value:
            value = value.replace("分", "")
            parts = value.split("H")
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            return hours * 60 + minutes
        elif "～" in value:
            parts = value.replace("分", "").split("～")
            return (int(parts[0]) + int(parts[1])) // 2
        elif "分" in value:
            return int(value.replace("分", ""))
    try:
        return int(value)
    except:
        return None


# 前処理済みデータを読み込む
df = pd.read_csv("../output/processed_data.csv", encoding="utf-8-sig")

# --- カラム変換処理を先に実行 ---
df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].apply(parse_distance)
df = df.dropna(subset=["最寄駅：距離（分）"])
df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].astype(int)

# ターゲット（取引価格）と特徴量に分ける
y = df["取引価格（総額）"]
X = df.drop("取引価格（総額）", axis=1)

# 保存
joblib.dump(X, "../output/X.pkl")
joblib.dump(y, "../output/y.pkl")

print("▼ Xとyをpkl形式で保存しました")
print("Xの形状:", X.shape)
print("yの形状:", y.shape)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 分割データを保存
joblib.dump(X_train, "../output/X_train.pkl")
joblib.dump(X_test, "../output/X_test.pkl")
joblib.dump(y_train, "../output/y_train.pkl")
joblib.dump(y_test, "../output/y_test.pkl")
