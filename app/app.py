import os
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from module.data_utils import load_and_clean_data

# Load environment
load_dotenv()
st.set_page_config(page_title="BI Assistant", layout="wide")
st.title("📊 Business Intelligence Workflow Assistant")

# File Upload
uploaded_file = st.file_uploader("📁 Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
        st.write("🧪 Temp file saved:", tmp_path)

    try:
        df = load_and_clean_data(tmp_path)
        st.success("✅ File loaded successfully!")
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

