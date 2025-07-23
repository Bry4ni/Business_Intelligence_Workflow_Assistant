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

# Add module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.data_utils import load_and_clean_data, infer_column_roles

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit UI setup
st.set_page_config(page_title="🌐 Multilingual BI Assistant", layout="wide")
st.title("🌐 Multilingual Business Intelligence Assistant")
st.markdown("Upload your dataset and ask a business question in any language.")

sns.set_theme(style="whitegrid")

# 🌍 Language options
language = st.sidebar.selectbox("🌍 Output Language", ["English", "Filipino", "Spanish", "Japanese", "Chinese"])
LANG_INSTRUCTION = {
    "English": "Respond in English.",
    "Filipino": "Isulat ang sagot sa Filipino.",
    "Spanish": "Responde en Español.",
    "Japanese": "日本語で回答してください。",
    "Chinese": "请用中文回答。"
}[language]

# 📁 Upload section
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        df = load_and_clean_data(tmp_path)
        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # 🔍 Column role inference
    try:
        inferred = infer_column_roles(df, os.getenv("GOOGLE_API_KEY"))
        st.markdown(f"🔍 **Inferred Columns:** {inferred}")
    except Exception as e:
        st.warning(f"⚠️ Could not infer column roles: {e}")
        inferred = {}

    # 📝 Prompt input
    user_prompt = st.text_area("📝 Enter your business question (any language):", height=120)

    if user_prompt.strip():
        schema = df.head(10).to_markdown(index=False)
        full_prompt = f"""
You are a multilingual business analyst.

{LANG_INSTRUCTION}

Based on the question:

**{user_prompt.strip()}**

Analyze this dataset and respond with:
1. Executive Summary
2. Python code for visualizations

Column Role Hints:
{inferred}

Sample Data:
{schema}
"""

        st.subheader("📤 Prompt Sent to Gemini")
        st.code(full_prompt)

        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(full_prompt)
            ai_text = response.text.strip()
        except Exception as e:
            st.error(f"❌ Gemini error: {e}")
            st.stop()

        import re

        # Extract summary before first code block
        summary_text = ai_text.split("```python")[0].strip()

        st.subheader("🧠 Executive Summary")
        st.markdown(summary_text)

        # Extract Python code blocks
        code_blocks = re.findall(r"```python(.*?)```", ai_text, re.DOTALL)

        st.subheader("📊 Visualizations")
        images = []

        for i, code in enumerate(code_blocks):
            # Safety: Skip code blocks that redefine df
            if "pd.DataFrame" in code or "data =" in code:
                st.warning(f"⚠️ Skipping generated DataFrame in Chart {i+1}")
                continue

            try:
                fig = plt.figure(figsize=(10, 6))
                local_vars = {"df": df.copy(), "plt": plt, "sns": sns}

                # Execute only plotting instructions
                exec(code, {}, local_vars)

                # Save and render chart
                tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmp_img.name, bbox_inches='tight')
                st.image(tmp_img.name, caption=f"Chart {i+1}")
                images.append((tmp_img.name, f"Chart {i+1}"))
            except Exception as e:
                st.error(f"⚠️ Chart {i+1} failed to render")
                st.exception(e)


        # 📄 Export to PDF
        st.subheader("📄 Export Summary + Charts to PDF")
        if st.button("⬇️ Download Report as PDF"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Business Intelligence Report", ln=True)
            pdf.set_font("Arial", '', 12)

            pdf.multi_cell(0, 8, ''.join(c for c in summary_text if ord(c) < 128))

            for path, title in images:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, title, ln=True)
                pdf.image(path, w=180)

            pdf_path = os.path.join(tempfile.gettempdir(), "bi_report_prompt_driven.pdf")
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF", f, "bi_report.pdf", mime="application/pdf")
