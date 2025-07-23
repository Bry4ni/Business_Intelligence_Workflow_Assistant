import os
import sys
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Set up module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.data_utils import load_and_clean_data, infer_column_roles

# Load environment and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit UI Setup
st.set_page_config(page_title="🌐 Multilingual BI Assistant", layout="wide")
st.title("📊 Multilingual Business Intelligence Assistant")

# 🌍 Language Selection
language = st.sidebar.selectbox("🌍 Select Output Language", ["English", "Filipino", "Spanish", "Japanese", "Chinese"])
LANG_INSTRUCTION = {
    "English": "Respond in English.",
    "Filipino": "Isulat ang sagot sa Filipino.",
    "Spanish": "Responde en Español.",
    "Japanese": "日本語で回答してください。",
    "Chinese": "请用中文回答。"
}[language]

# 📁 Upload Section
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        df = load_and_clean_data(tmp_path)
        st.subheader("📋 Preview Data")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()

    # 🔍 Column Role Inference
    try:
        inferred_roles = infer_column_roles(df, os.getenv("GOOGLE_API_KEY"))
        st.markdown(f"🔍 **Inferred Roles:** `{inferred_roles}`")
    except Exception as e:
        st.warning(f"⚠️ Could not infer column roles: {e}")
        inferred_roles = {}

    # 📝 Prompt
    user_prompt = st.text_area("📝 Enter your business question:", height=120)

    if user_prompt.strip():
        try:
            sample_data = df.head(10).to_dict(orient="records")
            schema_preview = json.dumps(sample_data, indent=2)

            full_prompt = f"""
You are a multilingual business analyst.

{LANG_INSTRUCTION}

Using the sample data and column role hints below, analyze the uploaded dataset and respond in JSON format.

**Question**: {user_prompt.strip()}

Respond in this JSON structure:
{{
  "summary": "...",
  "charts": [
    {{
      "chart_type": "bar" | "line" | "pie",
      "x": "...",
      "y": "...",
      "hue": "...", (optional)
      "title": "..."
    }}
  ]
}}

Column Role Hints: {inferred_roles}

Sample Data:
{schema_preview}
"""

            st.subheader("📤 Prompt Sent to Gemini")
            st.code(full_prompt)

            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            response = model.generate_content(full_prompt)
            result = json.loads(response.text)

            summary_text = result.get("summary", "")
            chart_instructions = result.get("charts", [])
        except Exception as e:
            st.error(f"❌ Could not parse Gemini response: {e}")
            st.text(response.text)
            st.stop()

        # 🧠 Summary
        st.subheader("🧠 Executive Summary")
        st.markdown(summary_text)

        # 📊 Chart Rendering
        st.subheader("📊 Visualizations")
        chart_paths = []

        for i, chart in enumerate(chart_instructions):
            chart_type = chart.get("chart_type")
            x = chart.get("x")
            y = chart.get("y")
            hue = chart.get("hue")
            title = chart.get("title", f"Chart {i+1}")

            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                if chart_type == "bar":
                    sns.barplot(data=df, x=x, y=y, hue=hue if hue else None, ax=ax)
                elif chart_type == "line":
                    sns.lineplot(data=df, x=x, y=y, hue=hue if hue else None, ax=ax)
                elif chart_type == "pie":
                    df.groupby(x)[y].sum().plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
                    ax.set_ylabel("")
                else:
                    st.warning(f"⚠️ Unsupported chart type: {chart_type}")
                    continue

                ax.set_title(title)
                chart_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(chart_file.name, bbox_inches="tight")
                st.image(chart_file.name, caption=title)
                chart_paths.append((chart_file.name, title))
            except Exception as e:
                st.error(f"⚠️ Failed to render chart {i+1}: {e}")

        # 📄 Export to PDF
        st.subheader("📄 Export Summary + Charts to PDF")
        if st.button("⬇️ Download Report as PDF"):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "Business Intelligence Report", ln=True)
                pdf.set_font("Arial", '', 12)

                clean_summary = ''.join(c for c in summary_text if ord(c) < 128)
                pdf.multi_cell(0, 8, clean_summary)

                for path, title in chart_paths:
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 10, title, ln=True)
                    pdf.image(path, w=180)

                pdf_path = os.path.join(tempfile.gettempdir(), "bi_report_prompt_driven.pdf")
                pdf.output(pdf_path)

                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download PDF", f, "bi_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"❌ Failed to generate PDF: {e}")