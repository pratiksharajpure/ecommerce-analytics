import streamlit as st

st.write("📁 Streamlit secrets path test")

try:
    db = st.secrets["database"]
    st.write("✅ Loaded secrets:", db)
except Exception as e:
    st.write("❌ Error loading secrets:", e)
