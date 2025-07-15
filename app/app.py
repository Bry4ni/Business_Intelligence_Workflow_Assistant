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
from module.data_utils import load_and_clean_data, find_best_column, infer_column_roles

# ------------------------------ Config
load_dotenv()
st.set_page_config(page_title="BI Assistant", layout="wide")
sns.set_theme(style="whitegrid", palette="Set2")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------------------ Title
st.title("🌐 Multilingual Business Intelligence Assistant")
st.markdown("Upload your spreadsheet in any language and receive AI-generated insights and charts.")

# ------------------------------ Language
language_choice = st.sidebar.selectbox("🌍 Select Output Language", ["English", "Filipino", "Spanish", "Japanese", "Chinese"])
LANG_INSTRUCTION = {
    "English": "Respond in English.",
    "Filipino": "Isulat ang sagot sa Filipino.",
    "Spanish": "Responde en Español.",
    "Japanese": "日本語で回答してください。",
    "Chinese": "请用中文回答。"
}[language_choice]

# ------------------------------ Column Mapping Helper
def safe_column(df, current, expected_role):
    if current and current in df.columns:
        return current
    if expected_role.lower() == "revenue":
        for col in df.select_dtypes(include='number').columns:
            return col
    elif expected_role.lower() == "month":
        for col in df.columns:
            if "month" in col.lower():
                return col
        if "Date" in df.columns:
            df["Month"] = pd.to_datetime(df["Date"], errors='coerce').dt.to_period("M")
            return "Month"
    else:
        for col in df.select_dtypes(include='object').columns:
            return col
    return None

# ------------------------------ Upload File
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # ✅ Use tempfile to handle uploaded files anywhere
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    df = load_and_clean_data(tmp_path)
    df.columns = [col.strip() for col in df.columns]
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # ------------------------------ Inference
    inferred = infer_column_roles(df, os.getenv("GOOGLE_API_KEY"))
    st.info("🔍 Gemini Column Mapping (raw)")
    st.json(inferred)

    revenue_col = safe_column(df, find_best_column(df, inferred.get("Revenue") or "Revenue"), "Revenue")
    product_col = safe_column(df, find_best_column(df, inferred.get("Product") or "Product"), "Product")
    region_col  = safe_column(df, find_best_column(df, inferred.get("Region") or "Region"), "Region")
    month_col   = safe_column(df, find_best_column(df, inferred.get("Month") or "Month"), "Month")

    st.markdown("### 🧭 Final Role Mapping")
    st.write({
        "Revenue": revenue_col,
        "Product": product_col,
        "Region": region_col,
        "Month": month_col
    })

    try:
        st.markdown("### 📊 Preview of Mapped Columns")
        st.dataframe(df[[revenue_col, product_col, region_col, month_col]].head())
    except:
        st.warning("⚠️ Could not preview mapped columns.")

    # ------------------------------ Executive Summary
    try:
        try:
            sample_preview = df.head(10).to_markdown(index=False)
        except Exception:
            sample_preview = df.head(10).to_string(index=False)

        schema_desc = "Columns: " + ", ".join(df.columns) + "\n\nSample Data:\n" + sample_preview
        full_prompt = f"""
Analyze the following dataset and provide a business-oriented summary with trends, patterns, and recommendations.

{LANG_INSTRUCTION}

{schema_desc}
"""
        st.subheader("📤 Prompt Sent to Gemini")
        st.code(full_prompt)

        st.subheader("🧠 Executive Summary")
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(full_prompt)
        ai_summary = response.text.strip()
        st.markdown(ai_summary)
    except Exception as e:
        st.error("❌ Gemini failed to generate insights.")
        st.code(str(e))
        ai_summary = ""

    # ------------------------------ Visualizations
    st.subheader("📈 Visualizations")
    if revenue_col in df.columns:
        df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')
    df_clean = df.dropna(subset=[revenue_col]) if revenue_col else df

    # Chart 1: Revenue by Product
    if revenue_col and product_col and all(c in df_clean.columns for c in [revenue_col, product_col]):
        grouped = df_clean.groupby(product_col)[revenue_col].sum().reset_index()
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=grouped, x=product_col, y=revenue_col, ax=ax1)
        ax1.set_title("Total Revenue per Product")
        st.pyplot(fig1)
    else:
        st.warning("⚠️ Skipping 'Revenue by Product' — columns not found or mapped.")

    # Chart 2: Monthly Revenue Trend
    if revenue_col and month_col and all(c in df_clean.columns for c in [revenue_col, month_col]):
        grouped = df_clean.groupby(month_col)[revenue_col].sum().reset_index()
        grouped[month_col] = grouped[month_col].astype(str)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=grouped, x=month_col, y=revenue_col, marker="o", ax=ax2)
        ax2.set_title("Monthly Revenue Trend")
        st.pyplot(fig2)
    else:
        st.warning("⚠️ Skipping 'Monthly Revenue Trend' — columns not found or mapped.")

    # Chart 3: Revenue by Region Pie Chart
    if revenue_col and region_col and all(c in df_clean.columns for c in [revenue_col, region_col]):
        grouped = df_clean.groupby(region_col)[revenue_col].sum()
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        grouped.plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax3)
        ax3.set_ylabel("")
        ax3.set_title("Revenue by Region")
        st.pyplot(fig3)
    else:
        st.warning("⚠️ Skipping 'Revenue by Region' — columns not found or mapped.")

    # ------------------------------ PDF Report Export
    st.subheader("📝 Export Report as PDF")

    with st.spinner("📄 Generating PDF Report..."):
        chart_paths = []

        try:
            grouped = df_clean.groupby(product_col)[revenue_col].sum().reset_index()
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(data=grouped, x=product_col, y=revenue_col, ax=ax1)
            ax1.set_title("Total Revenue per Product")
            img1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            fig1.savefig(img1.name, bbox_inches="tight")
            chart_paths.append((img1.name, "Total Revenue per Product"))
        except Exception as e:
            st.warning("⚠️ Could not render Chart 1")

        try:
            grouped = df_clean.groupby(month_col)[revenue_col].sum().reset_index()
            grouped[month_col] = grouped[month_col].astype(str)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=grouped, x=month_col, y=revenue_col, marker="o", ax=ax2)
            ax2.set_title("Monthly Revenue Trend")
            img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            fig2.savefig(img2.name, bbox_inches="tight")
            chart_paths.append((img2.name, "Monthly Revenue Trend"))
        except Exception as e:
            st.warning("⚠️ Could not render Chart 2")

        try:
            grouped = df_clean.groupby(region_col)[revenue_col].sum()
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            grouped.plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax3)
            ax3.set_ylabel("")
            ax3.set_title("Revenue by Region")
            img3 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            fig3.savefig(img3.name, bbox_inches="tight")
            chart_paths.append((img3.name, "Revenue by Region"))
        except Exception as e:
            st.warning("⚠️ Could not render Chart 3")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Business Intelligence Report", ln=True)
        pdf.set_font("Arial", '', 12)
        summary_clean = ''.join(c for c in ai_summary if ord(c) < 128)
        pdf.multi_cell(0, 8, summary_clean)

        for path, title in chart_paths:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, title, ln=True)
            pdf.image(path, w=180)

        pdf_path = os.path.join(tempfile.gettempdir(), "bi_report.pdf")
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Full Report (PDF)",
                data=f,
                file_name="bi_report.pdf",
                mime="application/pdf"
            )

else:
    st.info("📁 Please upload a CSV or Excel file to get started.")

