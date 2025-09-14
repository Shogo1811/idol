# === 5.3: Predicted Price を int に変換 ===
import joblib
import pandas as pd

# データ再作成（※ここでは一貫性を保つため再読み込み）
y_test = joblib.load("../output/y_test.pkl")
y_pred = joblib.load("../output/y_pred.pkl")

comparison = pd.DataFrame({"Actual Price": y_test.values, "Predicted Price": y_pred})

# 小数から整数へ（見やすさ重視）
comparison["Predicted Price"] = comparison["Predicted Price"].astype(int)

# 確認出力
print("✅ Predicted Price を整数に変換しました")
print(comparison.head())
