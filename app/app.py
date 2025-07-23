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
import json

# Add module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.data_utils import load_and_clean_data, infer_column_roles

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit config
st.set_page_config(page_title="📊 Multilingual BI Assistant", layout="wide")
st.title("🌐 Multilingual Business Intelligence Assistant")

# Theme
sns.set_theme(style="whitegrid")

# Language settings
language = st.sidebar.selectbox("🌍 Output Language", ["English", "Filipino", "Spanish", "Japanese", "Chinese"])
LANG_INSTRUCTION = {
    "English": "Respond in English.",
    "Filipino": "Isulat ang sagot sa Filipino.",
    "Spanish": "Responde en Español.",
    "Japanese": "日本語で回答してください。",
    "Chinese": "请用中文回答。"
}[language]

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        df = load_and_clean_data(tmp_path)
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"❌ File error: {e}")
        st.stop()

    # Role inference
    try:
        inferred_roles = infer_column_roles(df, os.getenv("GOOGLE_API_KEY"))
        st.markdown(f"🔍 **Inferred Roles:** {inferred_roles}")
    except Exception as e:
        st.warning(f"⚠️ Role inference failed: {e}")
        inferred_roles = {}

    # Prompt input
    user_prompt = st.text_area("📝 Enter your business question:", height=120)

    if user_prompt.strip():
        # Format sample safely (exclude un-serializable Timestamps)
        try:
            sample_data = df.head(10).copy()
            sample_data = sample_data.astype(str)
            sample = sample_data.to_dict(orient="records")
        except Exception as e:
            st.error(f"❌ Sample formatting error: {e}")
            st.stop()

        # Prompt Gemini
        full_prompt = f"""
You are a multilingual business analyst.

{LANG_INSTRUCTION}

Based on the question:

**{user_prompt.strip()}**

Analyze this dataset and respond with a JSON containing:
1. "summary": Summary insights.
2. "charts": List of chart instructions with "chart_type", "x", "y", optional "hue" and "title".

Inferred Column Roles:
{json.dumps(inferred_roles, indent=2)}

Sample Data:
{json.dumps(sample, indent=2)}
"""

        st.subheader("📤 Prompt Sent to Gemini")
        st.code(full_prompt)

        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(full_prompt)
            gemini_text = response.text.strip()

            # Try parsing response
            gemini_json = json.loads(gemini_text)
            summary_text = gemini_json.get("summary", "No summary returned.")
            chart_specs = gemini_json.get("charts", [])
        except Exception as e:
            st.error("❌ Could not parse Gemini response.")
            st.text(gemini_text)  # fallback: show raw Gemini response
            st.stop()

        # Display summary
        st.subheader("🧠 Executive Summary")
        st.markdown(summary_text)

        # Generate and show charts
        st.subheader("📊 Visualizations")
        images = []
        for i, chart in enumerate(chart_specs):
            try:
                fig = plt.figure(figsize=(10, 6))
                chart_type = chart.get("chart_type", "").lower()
                x = chart.get("x")
                y = chart.get("y")
                hue = chart.get("hue", None)
                title = chart.get("title", f"Chart {i+1}")

                if chart_type == "bar":
                    sns.barplot(data=df, x=x, y=y, hue=hue, ci=None)
                elif chart_type == "line":
                    sns.lineplot(data=df, x=x, y=y, hue=hue, ci=None)
                elif chart_type == "scatter":
                    sns.scatterplot(data=df, x=x, y=y, hue=hue)
                else:
                    st.warning(f"⚠️ Unsupported chart type: {chart_type}")
                    continue

                plt.title(title)
                plt.xticks(rotation=45)
                buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                plt.savefig(buf.name, bbox_inches="tight")
                st.image(buf.name)
                images.append((buf.name, title))
                plt.close()
            except Exception as e:
                st.error(f"⚠️ Chart {i+1} failed: {e}")

        # Export to PDF
        st.subheader("📄 Export to PDF")
        if st.button("⬇️ Download Report as PDF"):
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
                st.download_button("📥 Download PDF", f, "bi_report.pdf", mime="application/pdf")
