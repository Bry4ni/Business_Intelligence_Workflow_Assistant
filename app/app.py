# File: app/app.py

import os
import sys
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fpdf import FPDF
import google.generativeai as genai

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.data_utils import load_and_clean_data

# ----------------- Setup
load_dotenv()
st.set_page_config(page_title="🌍 Multilingual BI Assistant", layout="wide")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.title("🌐 Multilingual Business Intelligence Assistant")
st.markdown("Upload your spreadsheet and type your business question in any language.")

# ----------------- Upload Section
uploaded_file = st.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx"])
user_prompt = st.text_area("💬 Your Business Question (Any Language)", placeholder="e.g., Muéstrame las ventas mensuales por producto")

if uploaded_file and user_prompt.strip():
    file_ext = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        df = load_and_clean_data(tmp_path)
        df.columns = [col.strip() for col in df.columns]
        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # ----------------- Prompt for Gemini Summary
        schema_info = f"Columns: {', '.join(df.columns)}\n\nSample:\n{df.head(10).to_markdown(index=False)}"
        full_prompt = f"""
You are a business data analyst.

The user asks: "{user_prompt.strip()}"

Analyze the dataset and answer the user's question by generating an executive summary and corresponding visualizations.

{schema_info}

Respond in the language of the user prompt.
"""

        st.subheader("📤 Prompt Sent to Gemini")
        st.code(full_prompt)

        # ----------------- Get Executive Summary
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(full_prompt)
        summary = response.text.strip()
        st.subheader("🧠 Executive Summary")
        st.markdown(summary)

        # ----------------- Ask Gemini to Generate Visualization Code
        code_prompt = f"""
The user asked: "{user_prompt.strip()}"

Now write Python code using matplotlib or seaborn to visualize the dataset (already loaded as `df`). 
Use the column names shown in this markdown table:

{df.head(10).to_markdown(index=False)}

The dataset is already loaded as a pandas DataFrame named `df`. Return only valid Python code (no markdown).
Limit to 2 or 3 insightful plots only.
"""

        code_response = model.generate_content(code_prompt)
        viz_code = code_response.text.strip()

        st.subheader("📜 Gemini-Generated Visualization Code")
        st.code(viz_code, language="python")

        # ----------------- Safe Execution
        st.subheader("📊 Generated Visualizations")
        local_env = {'df': df, 'plt': plt, 'pd': pd}
        chart_paths = []
        try:
            exec(viz_code, local_env)
            # Automatically capture figures created
            for i, fig_num in enumerate(plt.get_fignums()):
                fig = plt.figure(fig_num)
                st.pyplot(fig)
                img_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                fig.savefig(img_path, bbox_inches="tight")
                chart_paths.append((img_path, f"Chart {i+1}"))
        except Exception as e:
            st.error("❌ Error running generated visualization code.")
            st.code(str(e))

        # ----------------- PDF Export
        if summary and chart_paths:
            st.subheader("📄 Export to PDF")

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Business Intelligence Summary", ln=True)
            pdf.set_font("Arial", '', 12)
            clean_summary = ''.join(c for c in summary if ord(c) < 128)  # Remove emojis or non-ASCII
            pdf.multi_cell(0, 8, clean_summary)

            for path, title in chart_paths:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, title, ln=True)
                pdf.image(path, w=180)

            pdf_path = os.path.join(tempfile.gettempdir(), "bi_custom_report.pdf")
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download Report as PDF", f, "business_report.pdf", mime="application/pdf")

    except Exception as e:
        st.error("❌ Failed to analyze the file.")
        st.code(str(e))
else:
    st.info("📌 Please upload a file and enter a prompt to begin.")


