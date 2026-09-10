from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CSV読み込み確認",
    page_icon="📄",
    layout="centered",
)

st.title("第4段階：監視銘柄CSVの読み込み確認")

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "assets" / "watchlist.csv"

st.write("読み込み対象：")
st.code(str(csv_path.relative_to(base_dir)))

if not csv_path.exists():
    st.error("assets/watchlist.csv が見つかりません。")
    st.stop()

try:
    # utf-8-sigは、Excelなどで保存されたCSVにも対応しやすい文字コードです
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

except UnicodeDecodeError:
    st.warning("UTF-8で読めなかったため、CP932で再試行します。")

    try:
        df = pd.read_csv(csv_path, encoding="cp932")
    except Exception as error:
        st.error("CSVの読み込みに失敗しました。")
        st.exception(error)
        st.stop()

except Exception as error:
    st.error("CSVの読み込みに失敗しました。")
    st.exception(error)
    st.stop()

if df.empty:
    st.error("CSVは読み込めましたが、データが空です。")
    st.stop()

st.success("CSVの読み込みに成功しました。")

st.write(f"データ行数：{len(df)} 行")

st.write("列名：")
st.code(", ".join(str(column) for column in df.columns))

st.write("先頭5行：")
st.dataframe(
    df.head(),
    use_container_width=True,
    hide_index=True,
)
