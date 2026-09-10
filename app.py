from pathlib import Path
import streamlit as st

app_dir = Path(__file__).resolve().parent

st.write("app.pyの場所:", str(app_dir))
st.write("同じフォルダ内のファイル:")
st.write([p.name for p in app_dir.iterdir()])

csv_files = list(app_dir.glob("*.csv"))
st.write("見つかったCSVファイル:")
st.write([p.name for p in csv_files])
