# === 5.2: Actual vs Predicted の DataFrame 作成 ===
import joblib
import pandas as pd

# データ読み込み
y_test = joblib.load("../output/y_test.pkl")
y_pred = joblib.load("../output/y_pred.pkl")

# 比較用 DataFrame を作成
comparison = pd.DataFrame({"Actual Price": y_test.values, "Predicted Price": y_pred})

# 確認出力
print("✅ comparison DataFrame 作成完了")
print(comparison.head())
