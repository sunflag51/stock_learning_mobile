from pathlib import Path
import csv

import pandas as pd
import streamlit as st


# ---------------------------------
# 1. Streamlit画面の基本設定
# ---------------------------------
st.set_page_config(
    page_title="ウォッチリスト確認",
    page_icon="📊",
    layout="wide",
)

st.title("ウォッチリスト確認")
st.write("CSVを読み込み、処理対象と処理対象外を確認します。")


# ---------------------------------
# 2. CSVファイルを作成
# ---------------------------------
csv_path = Path("assets/watchlist.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)

header = [
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

rows = [
    [
        1,
        "true",
        "任天堂",
        "7974" + "." + "JP",
        "7974" + "." + "T",
        "日本株",
        "ゲーム",
        "JPY",
        "操作確認用",
    ],
    [
        2,
        "true",
        "トヨタ自動車",
        "7203" + "." + "JP",
        "7203" + "." + "T",
        "日本株",
        "自動車",
        "JPY",
        "操作確認用",
    ],
    [
        3,
        "true",
        "Alphabet Class C",
        "GOOG" + "." + "US",
        "GOOG",
        "米国株",
        "情報技術",
        "USD",
        "操作確認用",
    ],
    [
        4,
        "true",
        "Tesla",
        "TSLA" + "." + "US",
        "TSLA",
        "米国株",
        "自動車",
        "USD",
        "操作確認用",
    ],
    [
        5,
        "false",
        "Apple",
        "AAPL" + "." + "US",
        "AAPL",
        "米国株",
        "情報技術",
        "USD",
        "無効化の確認用",
    ],
]

try:
    with csv_path.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)

    st.success("CSVファイルを正常に作成しました。")

except Exception as error:
    st.error("CSVファイルを作成できませんでした。")
    st.exception(error)
    st.stop()


# ---------------------------------
# 3. CSVを読み込む
# ---------------------------------
try:
    master_df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        dtype=str,
    )

except Exception as error:
    st.error("CSVファイルを読み込めませんでした。")
    st.exception(error)
    st.stop()


# ---------------------------------
# 4. 文字列の前後にある空白を削除
# ---------------------------------
for column in master_df.columns:
    master_df[column] = master_df[column].fillna("").str.strip()


# ---------------------------------
# 5. display_orderを数値に変換
# ---------------------------------
master_df["display_order"] = pd.to_numeric(
    master_df["display_order"],
    errors="coerce",
)

master_df = master_df.sort_values(
    by="display_order",
    ascending=True,
).reset_index(drop=True)


# ---------------------------------
# 6. enabledを判定用の真偽値に変換
# ---------------------------------
enabled_normalized = (
    master_df["enabled"]
    .str.strip()
    .str.lower()
)

master_df["enabled_bool"] = enabled_normalized.isin(
    ["true", "1", "yes", "on"]
)


# ---------------------------------
# 7. 全データを表示
# ---------------------------------
st.subheader("CSV全体")

display_columns = [
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

st.dataframe(
    master_df[display_columns],
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------
# 8. 処理対象と処理対象外に分ける
# ---------------------------------
target_df = master_df.loc[
    master_df["enabled_bool"]
].copy()

excluded_df = master_df.loc[
    ~master_df["enabled_bool"]
].copy()


# ---------------------------------
# 9. 件数を表示
# ---------------------------------
col1, col2, col3 = st.columns(3)

col1.metric(
    "CSV登録件数",
    len(master_df),
)

col2.metric(
    "処理対象",
    len(target_df),
)

col3.metric(
    "処理対象外",
    len(excluded_df),
)


# ---------------------------------
# 10. 処理対象を表示
# ---------------------------------
st.subheader("処理対象")

if target_df.empty:
    st.warning("処理対象がありません。")
else:
    st.dataframe(
        target_df[
            ["display_order", "name", "symbol", "provider_symbol"]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------
# 11. 処理対象外を表示
# ---------------------------------
st.subheader("処理対象外")

if excluded_df.empty:
    st.info("処理対象外の銘柄はありません。")
else:
    st.dataframe(
        excluded_df[
            ["display_order", "name", "symbol", "provider_symbol", "note"]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------
# 12. 最終確認
# ---------------------------------
expected_target_count = 4
expected_excluded_count = 1

if (
    len(target_df) == expected_target_count
    and len(excluded_df) == expected_excluded_count
):
    st.success(
        "確認完了：処理対象4件、処理対象外1件です。"
    )
else:
    st.error(
        "想定件数と一致しません。"
        f" 処理対象={len(target_df)}件、"
        f"処理対象外={len(excluded_df)}件です。"
    )
