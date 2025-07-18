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
st.title("🌐 Business Intelligence Assistant (Multilingual Prompt Mode)")
st.markdown("Upload your spreadsheet and enter a business question or analysis prompt in any language.")

# ------------------------------ Prompt Input
user_prompt = st.text_area("📝 Enter your business question or analysis prompt", 
                           placeholder="e.g. Analyze product trends and sales performance across all regions")

# ------------------------------ Upload
uploaded_file = st.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    # Load data
    df = load_and_clean_data(tmp_path)
    df.columns = [col.strip() for col in df.columns]

    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # ------------------------------ Column Inference
    inferred = infer_column_roles(df, os.getenv("GOOGLE_API_KEY"))
    st.info("🔍 Gemini Column Mapping")
    st.json(inferred)

    # Safe column resolution
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

    # ------------------------------ AI Summary from Prompt
    st.subheader("🧠 Gemini-Powered Insight")

    schema_desc = "Columns: " + ", ".join(df.columns) + "\n\nSample Data:\n" + df.head(10).to_markdown(index=False)

    full_prompt = f"""
You are a multilingual business analyst.

Dataset:
{schema_desc}

User Request:
{user_prompt or "Analyze the business performance and visualize key trends across products, regions, and months."}
"""

    st.subheader("📤 Prompt Sent to Gemini")
    st.code(full_prompt)

    try:
        model = genai.GenerativeModel("gemini-2.0-pro")
        response = model.generate_content(full_prompt)
        ai_summary = response.text.strip()
        st.markdown(ai_summary)
    except Exception as e:
        st.error("❌ Gemini failed to generate insights.")
        st.code(str(e))
        ai_summary = ""

    # ------------------------------ Visualizations
    st.subheader("📊 Visualizations")
    if revenue_col in df.columns:
        df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')
    df_clean = df.dropna(subset=[revenue_col]) if revenue_col else df

    def show_chart(title, fig):
        st.markdown(f"#### {title}")
        st.pyplot(fig)

    # Chart 1: Revenue by Product
    if revenue_col and product_col:
        grouped = df_clean.groupby(product_col)[revenue_col].sum().reset_index()
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=grouped, x=product_col, y=revenue_col, ax=ax1)
        ax1.set_title("Total Revenue per Product")
        show_chart("Product Revenue", fig1)

    # Chart 2: Monthly Trend
    if revenue_col and month_col:
        grouped = df_clean.groupby(month_col)[revenue_col].sum().reset_index()
        grouped[month_col] = grouped[month_col].astype(str)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=grouped, x=month_col, y=revenue_col, marker="o", ax=ax2)
        ax2.set_title("Monthly Revenue Trend")
        show_chart("Monthly Trend", fig2)

    # Chart 3: Revenue by Region
    if revenue_col and region_col:
        grouped = df_clean.groupby(region_col)[revenue_col].sum()
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        grouped.plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax3)
        ax3.set_ylabel("")
        ax3.set_title("Revenue by Region")
        show_chart("Region Pie Chart", fig3)

else:
    st.info("📁 Please upload a CSV or Excel file to get started.")


