from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="CSV保存場所確認",
    page_icon="📄",
    layout="centered",
)

st.title("第3段階：監視銘柄CSVの保存場所確認")

# app.pyが保存されているフォルダ
base_dir = Path(__file__).resolve().parent

# リポジトリ内にあるCSVファイルを検索
csv_files = sorted(base_dir.rglob("*.csv"))

if csv_files:
    st.success(f"CSVファイルが {len(csv_files)} 個見つかりました。")
    st.write("見つかったCSVファイル：")

    for csv_file in csv_files:
        relative_path = csv_file.relative_to(base_dir)
        st.code(str(relative_path))

else:
    st.error("CSVファイルが見つかりませんでした。")
    st.write("GitHub内にCSVファイルが保存されているか確認してください。")
