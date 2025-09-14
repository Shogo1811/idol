# === 4.4: モデルの評価（RMSEとR²） ===
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# テスト用の正解データと予測結果を読み込み
y_test = joblib.load("../output/y_test.pkl")
y_pred = joblib.load("../output/y_pred.pkl")

# RMSE（Root Mean Squared Error）を計算
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# R²スコアを計算
r2 = r2_score(y_test, y_pred)

# 評価結果を出力
print("✅ モデルの評価結果:")
print(f"RMSE: {rmse:.2f}")
print(f"R² score: {r2:.3f}")
