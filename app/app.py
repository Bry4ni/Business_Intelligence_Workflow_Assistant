# app/app.py

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.data_utils import load_and_clean_data, infer_column_roles

# Load environment
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
st.set_page_config(page_title="BI Assistant", layout="wide")
st.title("🌐 Multilingual BI Assistant")
st.markdown("Upload your dataset and ask a question in **any language**.")

# 🌍 Language options
language = st.sidebar.selectbox("🌍 Output Language", ["English", "Filipino", "Spanish", "Japanese", "Chinese"])
LANG_INSTRUCTION = {
    "English": "Respond in English.",
    "Filipino": "Isulat ang sagot sa Filipino.",
    "Spanish": "Responde en Español.",
    "Japanese": "日本語で回答してください。",
    "Chinese": "请用中文回答。"
}[language]

uploaded_file = st.file_uploader("📁 Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        df = load_and_clean_data(tmp_path)
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    user_prompt = st.text_area("📝 Enter your business question (any language):", height=120)
    if user_prompt.strip():
        with st.spinner("Analyzing with Gemini..."):
            inferred_roles = infer_column_roles(df, os.getenv("GOOGLE_API_KEY"))

            # Prompt Gemini to suggest chart type
            prompt = f"""
{LANG_INSTRUCTION}

The user asked: "{user_prompt.strip()}"

Here are the columns:
{list(df.columns)}

Suggest a good chart type based on the user's question and the inferred roles: {inferred_roles}.

Respond in JSON with:
{{
  "chart_type": "bar/line/pie/area",
  "summary": "Short summary of insights"
}}
"""
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)

            import json
            try:
                result = json.loads(response.text.strip())
                chart_type = result.get("chart_type", "bar").lower()
                summary_text = result.get("summary", "No summary returned.")
            except Exception:
                st.warning("⚠️ Could not parse Gemini output. Showing raw response.")
                chart_type = "bar"
                summary_text = response.text

        st.subheader("🧠 Executive Summary")
        st.markdown(summary_text)

        # Visualization
        st.subheader("📊 Visualization")
        fig, ax = plt.subplots()

        revenue = inferred_roles.get("Revenue")
        product = inferred_roles.get("Product")
        region = inferred_roles.get("Region")
        month = inferred_roles.get("Month")

        try:
            if chart_type == "bar" and product and revenue:
                df.groupby(product)[revenue].sum().plot(kind="bar", ax=ax)
            elif chart_type == "line" and month and revenue:
                df.groupby(month)[revenue].sum().plot(kind="line", ax=ax)
            elif chart_type == "pie" and region and revenue:
                df.groupby(region)[revenue].sum().plot(kind="pie", ax=ax, autopct="%1.1f%%")
            elif chart_type == "area" and month and revenue:
                df.groupby(month)[revenue].sum().plot(kind="area", ax=ax)
            else:
                st.warning("⚠️ Insufficient data for suggested chart.")
                ax.set_visible(False)

            if ax.has_data():
                st.pyplot(fig)
                chart_path = os.path.join(tempfile.gettempdir(), "chart.png")
                fig.savefig(chart_path)
            else:
                chart_path = None
        except Exception as e:
            st.error(f"❌ Chart rendering failed: {e}")
            chart_path = None

        # PDF Export
        try:
            st.subheader("📄 Export to PDF")
            if st.button("⬇️ Download Report as PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "Business Intelligence Report", ln=True)
                pdf.set_font("Arial", '', 12)
                pdf.multi_cell(0, 10, summary_text)

                if chart_path:
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 10, "Chart", ln=True)
                    pdf.image(chart_path, w=180)

                pdf_path = os.path.join(tempfile.gettempdir(), "bi_report.pdf")
                pdf.output(pdf_path)

                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download PDF", f, "bi_report.pdf", mime="application/pdf")

        except Exception as e:
            st.error("❌ Failed to process the file.")
            st.exception(e)
else:
    st.info("📂 Please upload a file and enter a business question.")
