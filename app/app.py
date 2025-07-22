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
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.data_utils import load_and_clean_data, infer_column_roles

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="BI Assistant", layout="wide")
st.title("📊 Multilingual Business Intelligence Assistant")

# 🌍 Language selector
language = st.sidebar.selectbox("🌐 Response Language", ["English", "Filipino", "Spanish", "Japanese", "Chinese"])
LANG_INSTRUCTION = {
    "English": "Respond in English.",
    "Filipino": "Isulat ang sagot sa Filipino.",
    "Spanish": "Responde en Español.",
    "Japanese": "日本語で回答してください。",
    "Chinese": "请用中文回答。"
}[language]

# 📁 File upload
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        df = load_and_clean_data(tmp_path)
        st.subheader("📄 Preview of Uploaded Data")
        st.dataframe(df.head(20), use_container_width=True)

        # Step 1: User prompt
        user_prompt = st.text_area("📝 Enter your business question (any language):", height=120)

        if user_prompt.strip():
            # Step 2: Role Inference
            inferred = infer_column_roles(df, os.getenv("GOOGLE_API_KEY"))
            revenue_col = inferred.get("Revenue")
            product_col = inferred.get("Product")
            region_col = inferred.get("Region")
            month_col = inferred.get("Month")

            # Step 3: Ask Gemini what chart to use
            prompt = f"""
{LANG_INSTRUCTION}

You are a business data analyst. Based on the prompt below and the data, suggest a suitable chart type and summarize insights.

Respond in this JSON format:
{{
  "chart_type": "<bar|line|pie|area>",
  "summary": "<short executive summary>"
}}

Prompt: {user_prompt}
Data columns: {', '.join(df.columns)}
"""
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            try:
                parsed = json.loads(response.text.strip())
                chart_type = parsed.get("chart_type", "").lower()
                summary_text = parsed.get("summary", "")
            except:
                st.warning("⚠️ Could not parse Gemini output. Showing raw response.")
                st.markdown(response.text)
                chart_type = None
                summary_text = ""

            st.subheader("🧠 Executive Summary")
            st.markdown(summary_text or "No summary generated.")

            # Step 4: Render chart based on inferred roles
            st.subheader("📈 Visualization")
            fig, chart_path = None, None

            try:
                plt.figure(figsize=(10, 6))
                if chart_type == "bar" and product_col and revenue_col:
                    chart = df.groupby(product_col)[revenue_col].sum().sort_values().plot(kind="barh")
                    plt.title(f"{revenue_col} by {product_col}")
                elif chart_type == "line" and month_col and revenue_col:
                    df_grouped = df.groupby(month_col)[revenue_col].sum().reset_index()
                    sns.lineplot(data=df_grouped, x=month_col, y=revenue_col, marker="o")
                    plt.xticks(rotation=45)
                    plt.title(f"{revenue_col} over {month_col}")
                elif chart_type == "pie" and region_col and revenue_col:
                    df_grouped = df.groupby(region_col)[revenue_col].sum()
                    df_grouped.plot(kind="pie", autopct='%1.1f%%', ylabel="")
                    plt.title(f"{revenue_col} by {region_col}")
                elif chart_type == "area" and month_col and product_col and revenue_col:
                    df_grouped = df.groupby([month_col, product_col])[revenue_col].sum().unstack(fill_value=0)
                    df_grouped.plot(kind="area", stacked=True)
                    plt.title(f"{revenue_col} by {product_col} over {month_col}")
                else:
                    st.warning("⚠️ Insufficient data for suggested chart.")
                    plt.close()
                    chart_type = None

                if chart_type:
                    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    plt.tight_layout()
                    plt.savefig(buf.name)
                    st.image(buf.name)
                    chart_path = buf.name
                plt.close()
            except Exception as e:
                st.error("⚠️ Failed to render chart.")
                st.exception(e)

            # Step 5: Export to PDF
            st.subheader("📄 Export to PDF")
            if st.button("⬇️ Download Report as PDF"):
                try:
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.add_page()
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
                    st.error("❌ PDF export failed.")
                    st.exception(e)
    except Exception as e:
        st.error("❌ Error reading file.")
        st.exception(e)
else:
    st.info("📂 Please upload a dataset to begin.")
