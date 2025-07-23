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

# Local import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.data_utils import load_and_clean_data, infer_column_roles

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="BI Assistant", layout="wide")
st.title("📊 Multilingual BI Assistant")
st.markdown("Upload your dataset and ask any business question in any language.")

# Language selection
language = st.sidebar.selectbox("🌍 Output Language", ["English", "Filipino", "Spanish", "Japanese", "Chinese"])
LANG_INSTRUCTION = {
    "English": "Respond in English.",
    "Filipino": "Isulat ang sagot sa Filipino.",
    "Spanish": "Responde en Español.",
    "Japanese": "日本語で回答してください。",
    "Chinese": "请用中文回答。"
}[language]

# File Upload
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        df = load_and_clean_data(tmp_path)
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"❌ File error: {e}")
        st.stop()

    # Inference of column roles
    try:
        inferred = infer_column_roles(df, os.getenv("GOOGLE_API_KEY"))
        st.markdown(f"🔍 **Inferred Roles:** `{inferred}`")
    except Exception as e:
        st.warning(f"⚠️ Could not infer column roles: {e}")
        inferred = {}

    # User prompt
    user_prompt = st.text_area("📝 Enter your business question:", height=120)

    if user_prompt.strip():
        schema = df.head(10).to_markdown(index=False)
        full_prompt = f"""
You are a multilingual business analyst.

{LANG_INSTRUCTION}

Based on the question:

**{user_prompt.strip()}**

Return only a JSON with this structure (no explanations):

{{
  "summary": "... executive summary ...",
  "charts": [
    {{
      "chart_type": "bar" | "line" | "pie" | "area",
      "x": "column_name",
      "y": "column_name",
      "hue": "optional_grouping_column",
      "title": "Chart Title"
    }}
  ]
}}

Column Hints: {inferred}

Sample Data:
{schema}
"""

        with st.spinner("🔍 Analyzing with Gemini..."):
            model = genai.GenerativeModel("gemini-2.0-pro")
            response = model.generate_content(full_prompt)
            try:
                result = json.loads(response.text.strip())
                summary_text = result["summary"]
                charts = result["charts"]
            except Exception as e:
                st.error("⚠️ Could not parse Gemini response.")
                st.text(response.text)
                st.stop()

        st.subheader("🧠 Executive Summary")
        st.markdown(summary_text)

        # Render charts
        st.subheader("📊 Visualizations")
        image_paths = []
        for i, chart in enumerate(charts):
            try:
                fig = plt.figure()
                chart_type = chart.get("chart_type")
                x = chart.get("x")
                y = chart.get("y")
                hue = chart.get("hue")
                title = chart.get("title", f"Chart {i+1}")

                if chart_type == "bar":
                    sns.barplot(data=df, x=x, y=y, hue=hue)
                elif chart_type == "line":
                    sns.lineplot(data=df, x=x, y=y, hue=hue)
                elif chart_type == "area":
                    df_sorted = df.sort_values(by=x)
                    df_sorted.set_index(x, inplace=True)
                    df_sorted[y].plot.area()
                elif chart_type == "pie":
                    pie_data = df.groupby(x)[y].sum()
                    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
                else:
                    st.warning(f"⚠️ Unknown chart type: {chart_type}")
                    continue

                plt.title(title)
                plt.tight_layout()
                img_path = os.path.join(tempfile.gettempdir(), f"chart_{i}.png")
                fig.savefig(img_path)
                image_paths.append((img_path, title))
                st.image(img_path, caption=title)
                plt.close(fig)
            except Exception as e:
                st.error(f"⚠️ Chart {i+1} failed: {e}")

        # PDF Export
        st.subheader("📄 Export Report to PDF")
        if st.button("⬇️ Download Full Report"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Business Intelligence Report", ln=True)
            pdf.set_font("Arial", '', 12)
            clean_summary = ''.join(c for c in summary_text if ord(c) < 128)
            pdf.multi_cell(0, 10, clean_summary)

            for path, title in image_paths:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, title, ln=True)
                pdf.image(path, w=180)

            pdf_path = os.path.join(tempfile.gettempdir(), "bi_report.pdf")
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF", f, "bi_report.pdf", mime="application/pdf")

