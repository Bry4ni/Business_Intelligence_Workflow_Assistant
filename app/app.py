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

# Import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.data_utils import load_and_clean_data, infer_column_roles

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit app setup
st.set_page_config(page_title="BI Assistant", layout="wide")
st.title("🌐 Multilingual Business Intelligence Assistant")

# Language selection
language = st.sidebar.selectbox("🌍 Output Language", ["English", "Filipino", "Spanish", "Japanese", "Chinese"])
LANG_INSTRUCTION = {
    "English": "Respond in English.",
    "Filipino": "Isulat ang sagot sa Filipino.",
    "Spanish": "Responde en Español.",
    "Japanese": "日本語で回答してください。",
    "Chinese": "请用中文回答。"
}[language]

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        df = load_and_clean_data(tmp_path)
        st.subheader("📋 Uploaded Data")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"❌ File loading failed: {e}")
        st.stop()

    # Infer column roles
    try:
        roles = infer_column_roles(df, os.getenv("GOOGLE_API_KEY"))
        st.markdown(f"🔍 **Inferred Roles:** `{roles}`")
    except Exception as e:
        st.warning(f"⚠️ Role inference failed: {e}")
        roles = {}

    # Prompt input
    user_prompt = st.text_area("📝 Enter your business question:", height=140)
    if user_prompt.strip():
        st.markdown("🔍 **Analyzing with Gemini...**")

        sample = df.head(10).to_dict(orient="records")
        prompt = f"""
You are a multilingual data analyst.

{LANG_INSTRUCTION}

Based on the following user request:

**{user_prompt.strip()}**

Analyze the uploaded data and respond in this JSON format:

{{
  "summary": "Write an executive summary based on the data and question.",
  "charts": [
    {{
      "chart_type": "bar" | "line" | "pie",
      "x": "column name",
      "y": "column name (skip for pie)",
      "hue": "column name (optional)",
      "title": "title of the chart"
    }},
    ...
  ]
}}

Data Sample:
{json.dumps(sample, indent=2)}

Column Role Hints:
{json.dumps(roles, indent=2)}
"""

        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            response_text = response.text.strip()

            # Parse response JSON
            parsed = json.loads(response_text)
            summary_text = parsed.get("summary", "")
            chart_instructions = parsed.get("charts", [])
        except Exception as e:
            st.error("⚠️ Could not parse Gemini response.")
            st.code(response_text)
            st.stop()

        # Show summary
        st.subheader("🧠 Executive Summary")
        st.markdown(summary_text)

        # Render charts
        st.subheader("📊 Visualizations")
        images = []

        for i, chart in enumerate(chart_instructions):
            try:
                chart_type = chart.get("chart_type", "")
                x = chart.get("x")
                y = chart.get("y")
                hue = chart.get("hue")
                title = chart.get("title", f"Chart {i+1}")

                fig, ax = plt.subplots(figsize=(10, 6))
                if chart_type == "bar" and x and y:
                    sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax, ci=None)
                elif chart_type == "line" and x and y:
                    sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, ci=None)
                elif chart_type == "pie" and x:
                    counts = df[x].value_counts()
                    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
                    ax.axis('equal')
                else:
                    st.warning(f"⚠️ Unsupported or incomplete chart config: {chart}")
                    continue

                ax.set_title(title)
                buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fig.savefig(buf.name, bbox_inches="tight")
                st.image(buf.name)
                images.append((buf.name, title))
            except Exception as e:
                st.error(f"⚠️ Chart {i+1} failed: {e}")

        # Export PDF
        st.subheader("📄 Export to PDF")
        if st.button("⬇️ Download Report as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Business Intelligence Report", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 10, ''.join(c for c in summary_text if ord(c) < 128))

            for path, title in images:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, title, ln=True)
                pdf.image(path, w=180)

            pdf_path = os.path.join(tempfile.gettempdir(), "bi_report.pdf")
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF", f, "bi_report.pdf", mime="application/pdf")

