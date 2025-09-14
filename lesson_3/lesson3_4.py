# === 3.4: データの一部を表示して確認 ===
import joblib

X_train = joblib.load("../output/X_train.pkl")
y_train = joblib.load("../output/y_train.pkl")

# 最初の5行を表示
print("▼ X_train（先頭5行）")
print(X_train.head())

print("\n▼ y_train（先頭5行）")
print(y_train.head())
