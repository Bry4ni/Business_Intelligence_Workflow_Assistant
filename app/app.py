import os
import sys
import tempfile
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from fpdf import FPDF
import google.generativeai as genai

# Setup path for local module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.data_utils import (
    load_and_clean_data,
    infer_column_roles,
    normalize_column_name,
    clean_gemini_json
)

# Config
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
st.set_page_config(page_title="🌐 BI Assistant", layout="wide")
st.title("🌐 Multilingual Business Intelligence Assistant")
sns.set_theme(style="whitegrid")

# Language handling
language = st.sidebar.selectbox("🌍 Output Language", ["English", "Filipino", "Spanish", "Japanese", "Chinese"])
LANG_INSTRUCTION = {
    "English": "Respond in English.",
    "Filipino": "Isulat ang sagot sa Filipino.",
    "Spanish": "Responde en Español.",
    "Japanese": "日本語で回答してください。",
    "Chinese": "请用中文回答。"
}[language]

# File uploader
uploaded_file = st.file_uploader("📤 Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        df = load_and_clean_data(tmp_path)
        st.subheader("📋 Preview Data")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # Inferred roles
    try:
        inferred_roles = infer_column_roles(df, os.getenv("GOOGLE_API_KEY"))
        st.markdown(f"🔍 Inferred Roles: {inferred_roles}")
    except:
        st.warning("⚠️ Role inference failed.")
        inferred_roles = {}

    # Default general prompt
    default_general_prompt = "Analyze the following dataset and provide a business-oriented summary with trends, patterns, and recommendations."
    st.markdown(f"**Default Prompt**: _{default_general_prompt}_")

    # Sample prompt generator
    if st.button( "Generate"):
        st.markdown(" Generating...")
        prompt_generator = f"""
{LANG_INSTRUCTION}

Given this general instruction:
"{default_general_prompt}"

Generate business-specific prompts a user might ask about a dataset like sales, finance, marketing, or churn. 
Respond ONLY as a JSON list:
[
  "...",
  "...",
  "..."
]
"""
        try:
            response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt_generator)
            prompt_suggestions = json.loads(response.text.strip())
            if isinstance(prompt_suggestions, list):
                selected = st.selectbox(prompt_suggestions)
                if selected:
                    st.session_state["user_prompt"] = selected
        except Exception as e:
            st.error("❌ Could not generate or parse sample prompts.")

    # Prompt input area
    user_prompt = st.text_area("📝 Enter your business question:", height=100, value=st.session_state.get("user_prompt", default_general_prompt))

    if user_prompt.strip():
        st.markdown("🔍 Analyzing with Gemini...")
        try:
            sample_json = json.loads(df.head(10).to_json(orient="records", date_format="iso", force_ascii=False))
        except:
            st.error("❌ Could not convert data to JSON.")
            st.stop()

        analysis_prompt = f"""
{LANG_INSTRUCTION}

You are a multilingual business analyst.

Using the sample data and column role hints below, analyze the uploaded dataset and respond in JSON format.

**User Prompt**: {user_prompt.strip()}

Respond ONLY in this JSON format:
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

Column Role Hints: {json.dumps(inferred_roles)}
Sample Data: {json.dumps(sample_json, indent=2)}
"""
        # Gemini generation
        try:
            response = genai.GenerativeModel("gemini-2.0-flash").generate_content(analysis_prompt)
            parsed = clean_gemini_json(response.text.strip())
            summary = parsed["summary"]
            chart_instructions = parsed["charts"]
        except Exception as e:
            st.error(f"❌ Could not parse Gemini response.")
            st.code(response.text.strip())
            st.stop()

        # Summary
        st.subheader("🧠 Executive Summary")
        st.markdown(summary)

        # Charts
        st.subheader("📊 Visualizations")
        images = []
        for i, chart in enumerate(chart_instructions):
            try:
                chart_type = chart["chart_type"]
                x = normalize_column_name(chart["x"], df.columns)
                y = normalize_column_name(chart["y"], df.columns)
                hue = normalize_column_name(chart.get("hue"), df.columns)
                title = chart.get("title", f"Chart {i+1}")

                fig = plt.figure(figsize=(10, 6))
                if chart_type == "bar":
                    sns.barplot(data=df, x=x, y=y, hue=hue, estimator="sum", ci=None)
                elif chart_type == "line":
                    sns.lineplot(data=df, x=x, y=y, hue=hue, estimator="sum", ci=None)
                elif chart_type == "pie":
                    pie_data = df.groupby(x)[y].sum()
                    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
                else:
                    raise ValueError("Unsupported chart type")

                plt.title(title)
                plt.xticks(rotation=45)
                plt.tight_layout()
                img_path = os.path.join(tempfile.gettempdir(), f"chart_{i}.png")
                plt.savefig(img_path)
                st.image(img_path)
                images.append((img_path, title))

            except Exception as e:
                st.error(f"⚠️ Failed to render chart {i+1}: {e}")

        # PDF export
        st.subheader("📄 Export Report to PDF")
        if st.button("⬇️ Download Report as PDF"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Business Intelligence Report", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 10, summary)

            for path, title in images:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, title, ln=True)
                pdf.image(path, w=180)

            pdf_path = os.path.join(tempfile.gettempdir(), "bi_report.pdf")
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF", f, "bi_report.pdf", mime="application/pdf")
