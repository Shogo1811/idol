# === 4.1: ランダムフォレストモデルの準備 ===
import joblib
from sklearn.ensemble import RandomForestRegressor

# データの読み込み（学習用のみ）
X_train = joblib.load("../output/X_train.pkl")
y_train = joblib.load("../output/y_train.pkl")

# モデルの作成
model = RandomForestRegressor(random_state=42)

# モデルの内容を表示
print("モデルの構成:")
print(model)

# モデルを保存（以降のファイルで読み込む用）
joblib.dump(model, "../output/random_forest_model.pkl")
