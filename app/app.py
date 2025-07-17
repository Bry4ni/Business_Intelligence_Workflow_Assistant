# app/app.py

import os
import sys
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from fpdf import FPDF
import google.generativeai as genai
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.data_utils import load_and_clean_data

# Load environment

# Load environment
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
st.set_page_config(page_title="BI Assistant", layout="wide")

st.title("🌐 Multilingual BI Assistant")
st.markdown("Upload your dataset and enter a business question in **any language**.")

# 1. Upload File
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
user_prompt = st.text_area("📣 Enter your business question (any language):", placeholder="e.g. ¿Cuál es la tendencia de ventas por producto este año?")

if uploaded_file and user_prompt.strip():
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.write(uploaded_file.getbuffer())
    tmp_path = tmp_file.name

    try:
        df = load_and_clean_data(tmp_path)
        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # 2. Prompt Gemini
        sample = df.head(10).to_markdown(index=False)
        column_list = ", ".join(df.columns)
        full_prompt = f"""
You are a multilingual business analyst.

{LANG_INSTRUCTION}

Analyze the data below based on the following question:

**User Prompt**: {user_prompt}

Give:
1. Executive summary with trends and insights (in user's language).
2. Python code using matplotlib/seaborn to visualize relevant insights.

Columns: {column_list}
Sample data:
{sample}
"""

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(full_prompt)
        response_text = response.text.strip()

        # 3. Extract code + summary
        import re
        code_match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
        chart_code = code_match.group(1).strip() if code_match else None
        summary_text = re.sub(r"```python.*?```", "", response_text, flags=re.DOTALL).strip()

        # 4. Show summary
        st.subheader("🧠 Executive Summary")
        st.markdown(summary_text)

        # 5. Run and show chart(s)
        st.subheader("📈 Visualization")
        chart_paths = []
        if chart_code:
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_file:
                    # Setup local namespace
                    local_vars = {"df": df, "plt": plt, "sns": sns}
                    exec(chart_code, {}, local_vars)
                    plt.savefig(img_file.name)
                    st.image(img_file.name)
                    chart_paths.append(img_file.name)
            except Exception as e:
                st.error(f"⚠️ Chart {i+1} failed: {e}")

        # 6. PDF Export
        st.subheader("📄 Export PDF Report")
        if st.button("⬇️ Download Full Report as PDF"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "BI Report", ln=True)
            pdf.set_font("Arial", '', 12)

            clean_summary = ''.join(c for c in summary_text if ord(c) < 128)
            pdf.multi_cell(0, 8, clean_summary)

            for path, title in images:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, title, ln=True)
                pdf.image(path, w=180)

            pdf_path = os.path.join(tempfile.gettempdir(), "bi_report_prompt_driven.pdf")
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF", f, "bi_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error("❌ Failed to process the file.")
        st.exception(e)
else:
    st.info("📂 Please upload a file and enter a business question.")
