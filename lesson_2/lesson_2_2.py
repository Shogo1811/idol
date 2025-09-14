from datetime import datetime

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


# CSV読み込み（前処理ステップを踏んだ上で）
df = pd.read_csv("../input/Tokyo_20242_20242.csv", encoding="cp932")
df.columns = df.columns.str.replace("\ufeff", "")

# 必要な列に絞る
columns_to_use = [
    "取引価格（総額）",
    "面積（㎡）",
    "最寄駅：距離（分）",
    "建築年",
    "市区町村名",
]
df = df[columns_to_use].dropna()

# 整数型へ変換
df["建築年"] = df["建築年"].str.replace("年", "", regex=False)
df["建築年"] = pd.to_numeric(df["建築年"], errors="coerce")
df = df.dropna(subset=["建築年"])
df["建築年"] = df["建築年"].astype(int)


# # 整数型へ変換（例：距離や建築年）
df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].apply(parse_distance)

# 変換失敗した行（Noneが入った）を除外
df = df.dropna(subset=["最寄駅：距離（分）"])
df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].astype(int)


# === 築年数の計算 ===
current_year = datetime.now().year
df["築年数"] = current_year - df["建築年"]

# === 異常値の除去 ===
# 価格・面積が0やマイナスのものは除外
df = df[df["取引価格（総額）"] > 0]
df = df[df["面積（㎡）"] > 0]
df = df[df["築年数"] >= 0]  # 未来の建築年を排除

print("▼ 処理後のデータ（先頭5行）")
print(df.head())
