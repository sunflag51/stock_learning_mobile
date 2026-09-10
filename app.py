import streamlit as st

st.set_page_config(
    page_title="設定ファイル確認",
    page_icon="⚙️",
    layout="centered",
)

st.title("第2段階：設定ファイル確認")

try:
    import config.settings as settings

    st.success("config/settings.py の読み込みに成功しました。")

    st.write("設定ファイルの保存場所も正常です。")

except Exception as e:
    st.error("config/settings.py の読み込みに失敗しました。")
    st.exception(e)
