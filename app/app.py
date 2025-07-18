# app/app.py

import os
import sys
import tempfile
import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from fpdf import FPDF
import google.generativeai as genai

# Import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.data_utils import load_and_clean_data

# --- Load environment and configure Gemini ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Streamlit UI ---
st.set_page_config(page_title="Multilingual BI Assistant", layout="wide")
st.title("🌐 Multilingual Business Intelligence Assistant")
st.markdown("Upload your dataset and ask a business question in any language.")

sns.set_theme(style="whitegrid")

# --- Language selector ---
language = st.sidebar.selectbox("🌍 Output Language", ["English", "Filipino", "Spanish", "Japanese", "Chinese"])
LANG_INSTRUCTION = {
    "English": "Respond in English.",
    "Filipino": "Isulat ang sagot sa Filipino.",
    "Spanish": "Responde en Español.",
    "Japanese": "日本語で回答してください。",
    "Chinese": "请用中文回答。"
}[language]

# --- File Upload ---
uploaded_file = st.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        df = load_and_clean_data(tmp_path)
        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # --- Prompt after successful data load ---
    user_prompt = st.text_area("📝 Enter your business question (any language):", height=120)
    if user_prompt.strip():
        # Gemini Prompt Construction
        schema = "Columns:\n" + ", ".join(df.columns)
        sample = df.head(10).to_markdown(index=False)
        prompt = f"""
You are a multilingual business analyst.
{LANG_INSTRUCTION}

Analyze the data below based on this question:
**{user_prompt.strip()}**

Please respond with:
1. Executive Summary
2. Python code for relevant visualizations

{schema}

Sample Data:
{sample}
"""

        # Show prompt for transparency
        st.subheader("📤 Prompt Sent to Gemini")
        st.code(prompt)

        # AI response from Gemini
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            ai_text = response.text.strip()
        except Exception as e:
            st.error(f"❌ Gemini error: {e}")
            st.stop()

        st.subheader("🧠 Executive Summary + Code")
        st.markdown(ai_text)

        # Extract code blocks
        summary_text = ai_text.split("```python")[0].strip()
        code_blocks = re.findall(r"```python(.*?)```", ai_text, re.DOTALL)

        # Execute & display safe charts
        st.subheader("📊 Visualizations")
        images = []
        safe_keywords = ("df", "sns", "plt", "groupby", "sum", "mean", "count", "plot")

        for i, code in enumerate(code_blocks):
            if not any(k in code for k in safe_keywords):
                st.warning(f"⚠️ Skipping Chart {i+1}: Unsafe or unrelated code.")
                continue
            try:
                fig = plt.figure()
                local_vars = {"df": df.copy(), "sns": sns, "plt": plt}
                exec(code, {}, local_vars)
                buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fig.savefig(buf.name, bbox_inches="tight")
                st.image(buf.name)
                images.append((buf.name, f"Chart {i+1}"))
            except Exception as e:
                st.error(f"⚠️ Chart {i+1} failed: {e}")

        # Export to PDF
        st.subheader("📄 Export Summary + Charts to PDF")
        if st.button("Generate PDF Report"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Business Intelligence Report", ln=True)
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
                st.download_button("⬇️ Download Report (PDF)", data=f, file_name="bi_report.pdf", mime="application/pdf")



