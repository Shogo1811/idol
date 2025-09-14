import pandas as pd

# CSV読み込み（前ステップと同じくUTF-8 BOM対応）
file_path = "lesson_1.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig")
df.columns = df.columns.str.replace("\ufeff", "")  # 念のためBOM除去

# --- カラムの追加 ---
# 単価（円/㎡）列を追加（取引価格 ÷ 面積）
df["単価（円/㎡）"] = df["取引価格（総額）"] / df["面積（㎡）"]
print("▼ 単価（円/㎡）を追加")
print(df[["取引価格（総額）", "面積（㎡）", "単価（円/㎡）"]].head())

# --- カラムの削除 ---
# 「建築年」を一旦削除（axis=1で列方向を指定）
df = df.drop(columns=["建築年"])
print("\n▼ '建築年' を削除")
print(df.head())

# --- カラム名の変更 ---
df = df.rename(
    columns={"市区町村名": "エリア名", "最寄駅：距離（分）": "駅からの距離（分）"}
)
print("\n▼ カラム名を変更")
print(df.head())

# --- データ型の変換 ---
# float → int（小数点以下切り捨て）に変換
df["駅からの距離（分）"] = df["駅からの距離（分）"].astype(int)
print("\n▼ データ型をintに変換")
print(df.dtypes)

# --- 保存（任意） ---
df.to_csv("../output/lesson_1_3_output.csv", index=False, encoding="utf-8-sig")
