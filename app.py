import streamlit as st

st.set_page_config(
    page_title="動作確認",
    page_icon="✅",
    layout="centered",
)

st.title("第2回 動作確認")

st.success("Streamlitの画面表示に成功しました。")

st.write("この文字が表示されれば、第1段階は正常です。")
