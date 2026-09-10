from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 画面の基本設定
# --------------------------------------------------
st.set_page_config(
    page_title="株式学習アプリ",
    page_icon="📊",
    layout="wide",
)

st.title("📊 株式学習アプリ")
st.subheader("第9段階：取得データの検証・正規化")


# --------------------------------------------------
# CSVファイルの場所
# app.pyと同じ階層にあるassets/watchlist.csvを指定
# --------------------------------------------------
app_dir = Path(__file__).resolve().parent
csv_path = app_dir / "assets" / "watchlist.csv"


# --------------------------------------------------
# CSVファイルの存在確認
# --------------------------------------------------
st.write("### 1. CSVファイルの確認")
st.write("CSVパス：", str(csv_path))
st.write("CSVの存在確認：", csv_path.exists())

if not csv_path.exists():
    st.error(
        "CSVファイルが見つかりません。"
        "assetsフォルダ内にwatchlist.csvがあるか確認してください。"
    )
    st.stop()


# --------------------------------------------------
# CSVファイルの読み込み
# --------------------------------------------------
try:
    master_df = pd.read_csv(csv_path)
except Exception as error:
    st.error(f"CSVの読み込みに失敗しました：{error}")
    st.stop()


# --------------------------------------------------
# 読み込んだCSVを表示
# --------------------------------------------------
st.write("### 2. CSVの読み込み結果")
st.write(f"データ件数：{len(master_df)}件")
st.write(f"列数：{len(master_df.columns)}列")
st.dataframe(master_df, use_container_width=True)


# --------------------------------------------------
# 必須列の確認
# --------------------------------------------------
required_columns = [
    "display_order",
    "enabled",
    "name",
    "symbol",
    "provider_symbol",
    "market",
    "category",
    "currency",
    "note",
]

missing_columns = [
    column for column in required_columns
    if column not in master_df.columns
]

st.write("### 3. 必須列の確認")

if missing_columns:
    st.error(
        "必要な列が不足しています："
        + ", ".join(missing_columns)
    )
    st.stop()
else:
    st.success("必要な9列がすべて存在します。")


# --------------------------------------------------
# データの正規化
# --------------------------------------------------
normalized_df = master_df.copy()

# 文字列列の前後の空白を削除
text_columns = [
    "name",
    "symbol",
    "provider_symbol",
    "market",
    "category",
    "currency",
    "note",
]

for column in text_columns:
    normalized_df[column] = (
        normalized_df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

# enabled列をTrueまたはFalseに統一
normalized_df["enabled"] = (
    normalized_df["enabled"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    })
)

# display_order列を数値に変換
normalized_df["display_order"] = pd.to_numeric(
    normalized_df["display_order"],
    errors="coerce",
)


# --------------------------------------------------
# 欠損・変換失敗の確認
# --------------------------------------------------
st.write("### 4. データの検証")

validation_errors = []

if normalized_df["display_order"].isna().any():
    validation_errors.append(
        "display_order列に数値へ変換できない値があります。"
    )

if normalized_df["enabled"].isna().any():
    validation_errors.append(
        "enabled列にTrueまたはFalseへ変換できない値があります。"
    )

important_text_columns = [
    "name",
    "symbol",
    "provider_symbol",
    "market",
    "currency",
]

for column in important_text_columns:
    if normalized_df[column].eq("").any():
        validation_errors.append(
            f"{column}列に空欄があります。"
        )

if normalized_df["symbol"].duplicated().any():
    duplicated_symbols = (
        normalized_df.loc[
            normalized_df["symbol"].duplicated(keep=False),
            "symbol",
        ]
        .unique()
        .tolist()
    )
    validation_errors.append(
        "symbol列に重複があります："
        + ", ".join(duplicated_symbols)
    )

if validation_errors:
    for message in validation_errors:
        st.error(message)
    st.stop()
else:
    st.success("データの検証に問題はありません。")


# --------------------------------------------------
# 表示順に並べ替え
# --------------------------------------------------
normalized_df = (
    normalized_df
    .sort_values("display_order")
    .reset_index(drop=True)
)


# --------------------------------------------------
# enabledがTrueの銘柄だけを抽出
# --------------------------------------------------
target_master = (
    normalized_df.loc[
        normalized_df["enabled"] == True
    ]
    .copy()
    .reset_index(drop=True)
)

excluded_master = (
    normalized_df.loc[
        normalized_df["enabled"] == False
    ]
    .copy()
    .reset_index(drop=True)
)


# --------------------------------------------------
# 正規化後の結果を表示
# --------------------------------------------------
st.write("### 5. 正規化後の全データ")
st.dataframe(normalized_df, use_container_width=True)

st.write("### 6. 処理対象（enabled=true）")
st.write(f"処理対象：{len(target_master)}件")
st.dataframe(
    target_master[
        [
            "display_order",
            "name",
            "symbol",
            "provider_symbol",
            "market",
            "currency",
        ]
    ],
    use_container_width=True,
)

st.write("### 7. 処理対象外（enabled=false）")
st.write(f"処理対象外：{len(excluded_master)}件")
st.dataframe(
    excluded_master[
        [
            "display_order",
            "name",
            "symbol",
            "provider_symbol",
            "market",
            "currency",
        ]
    ],
    use_container_width=True,
)


# --------------------------------------------------
# 最終確認
# --------------------------------------------------
expected_target_symbols = {
    " 7974.JP ",
    " 7203.JP ",
    " GOOG.US ",
    " TSLA.US ",
}

actual_target_symbols = set(target_master["symbol"].tolist())

st.write("### 8. 最終確認")

if actual_target_symbols == expected_target_symbols:
    st.success(
        "確認完了：任天堂、トヨタ自動車、"
        "Alphabet Class C、Teslaの4件が処理対象です。"
    )
else:
    st.warning(
        "想定している4銘柄と処理対象が一致しません。"
        "上の表を確認してください。"
    )

if " AAPL.US " in excluded_master["symbol"].tolist():
    st.success("Apple（AAPL.US）は処理対象外になっています。")
else:
    st.warning(
        "Apple（AAPL.US）の処理対象外設定を確認してください。"
    )
