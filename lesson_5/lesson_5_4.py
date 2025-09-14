# === 5.4: comparison.csv を保存 ===
import joblib
import pandas as pd

# データ読み込み
y_test = joblib.load("../output/y_test.pkl")
y_pred = joblib.load("../output/y_pred.pkl")

# 比較データ作成
comparison = pd.DataFrame({"Actual Price": y_test.values, "Predicted Price": y_pred})

# 整数化
comparison["Predicted Price"] = comparison["Predicted Price"].astype(int)

# 保存
output_path = "../output/comparison.csv"
comparison.to_csv(output_path, index=False, encoding="utf-8-sig")

# 確認
print(f"✅ 保存完了: {output_path}")
print(comparison.head())
