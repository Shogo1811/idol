import pandas as pd

# CSVファイル読み込み（文字コードに注意）
file_path = "lesson_1.csv"
# df = pd.read_csv(file_path, encoding="cp932")  # Excel出力ならcp932が多い
df = pd.read_csv(file_path, encoding="utf-8-sig")


# DataFrameの基本情報確認
print(df.head())  # 先頭5件
print(df.tail())  # 最後5件
print(df.shape)  # 行数・列数
print(df.columns)  # カラム名
print(df.info())  # データ型など
print(df.describe())  # 統計量
