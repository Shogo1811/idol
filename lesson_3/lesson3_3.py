# === 3.3: データサイズの確認 ===
import joblib
import pandas as pd

X_train = joblib.load("../output/X_train.pkl")
X_test = joblib.load("../output/X_test.pkl")
y_train = joblib.load("../output/y_train.pkl")
y_test = joblib.load("../output/y_test.pkl")

# 形状を確認
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
