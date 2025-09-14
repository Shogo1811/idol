# filename: linear_rent_regression_simple.py
# 使い方:
#   python linear_rent_regression_simple.py
#
# 面積(m²) → 家賃(円) の単回帰モデル
# データは自動生成されるのでCSV不要！

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def main():
    # サンプルデータを自動生成
    rng = np.random.default_rng(42)
    areas = rng.uniform(15, 70, size=50)  # 15〜70m²
    rents = 30000 + 2200 * areas + rng.normal(0, 8000, size=areas.size)

    X = areas.reshape(-1, 1)  # 説明変数（2次元配列）
    y = rents  # 目的変数

    # 学習/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # モデル学習
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 係数と切片
    a = float(model.coef_[0])  # 傾き
    b = float(model.intercept_)  # 切片
    print("===== 学習結果 =====")
    print(f"回帰式: rent = {a:.1f} * area + {b:.0f}")

    # 評価
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R²: {r2:.3f}\n")

    # 可視化（散布図＋回帰直線）
    plt.scatter(X, y, label="data")
    xs = np.linspace(15, 70, 200).reshape(-1, 1)
    ys = model.predict(xs)
    plt.plot(xs, ys, color="red", label="fitted line")
    plt.xlabel("Area (m²)")
    plt.ylabel("Rent (yen)")
    plt.legend()
    plt.title("Linear Regression (Area → Rent)")
    plt.tight_layout()
    plt.savefig("rent_simple.png", dpi=150)
    print("🖼 グラフを rent_simple.png に保存しました")

    # サンプル予測
    demo = np.array([[30.0], [45.0]])
    demo_pred = model.predict(demo)
    for a_m2, pred in zip(demo.flatten(), demo_pred):
        print(f"面積 {a_m2:.0f} m² → 予測家賃 {pred:,.0f} 円")


if __name__ == "__main__":
    main()
