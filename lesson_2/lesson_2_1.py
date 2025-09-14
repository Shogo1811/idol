import pandas as pd


def parse_distance(value):
    if isinstance(value, str):
        value = value.strip()
        if "H" in value:
            # 例: "1H" or "1H30分"
            value = value.replace("分", "")
            parts = value.split("H")
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            return hours * 60 + minutes
        elif "～" in value:
            # 例: "30分～60分" → 平均を取る
            parts = value.replace("分", "").split("～")
            return (int(parts[0]) + int(parts[1])) // 2
        elif "分" in value:
            return int(value.replace("分", ""))
    try:
        return int(value)
    except:
        return None


file_path = "../input/Tokyo_20242_20242.csv"
df = pd.read_csv(file_path, encoding="cp932")
# df = pd.read_csv(file_path, encoding="utf-8-sig")
# df.columns = df.columns.str.replace("\ufeff", "")  # BOM除去

# 必要な列だけを選択（不要な列は除外）
columns_to_use = [
    "取引価格（総額）",
    "面積（㎡）",
    "最寄駅：距離（分）",
    "建築年",
    "市区町村名",
]
df = df[columns_to_use]

# 欠損値を含む行を削除（最低限の前処理）
df = df.dropna()

# print(df)

# # 整数型へ変換（例：距離や建築年）
df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].apply(parse_distance)

# 変換失敗した行（Noneが入った）を除外
df = df.dropna(subset=["最寄駅：距離（分）"])
df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].astype(int)

df["建築年"] = df["建築年"].str.replace("年", "", regex=False)
df["建築年"] = pd.to_numeric(df["建築年"], errors="coerce")
df = df.dropna(subset=["建築年"])
df["建築年"] = df["建築年"].astype(int)

# # データの確認
print(df.head())
