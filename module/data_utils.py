# module/data_utils.py

import pandas as pd
import os
import difflib
import chardet
import google.generativeai as genai

COLUMN_SYNONYMS = {
    "Revenue": ["Revenue", "Sales", "Total", "Income", "Ingresos", "Amount"],
    "Product": ["Product", "Item", "Producto", "Product_Name"],
    "Region": ["Region", "Area", "Territory", "Zone", "Región"],
    "Month": ["Month", "Mes", "Periodo", "Period"]
}

def load_and_clean_data(filepath):
    ext = os.path.splitext(filepath)[-1].lower()

    if ext in [".xls", ".xlsx"]:
        try:
            df = pd.read_excel(filepath, engine="openpyxl")
        except Exception as e:
            raise ValueError(f"❌ Excel read error: {e}")
    elif ext in [".csv", ".txt"]:
        try:
            # Auto-detect encoding with chardet
            with open(filepath, "rb") as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result["encoding"] or "utf-8"

            df = pd.read_csv(filepath, encoding=encoding)
        except Exception as e:
            raise ValueError(f"❌ CSV read error with encoding {encoding}: {e}")
    else:
        raise ValueError("❌ Unsupported file type. Please upload a .csv or .xlsx file.")

    if df.empty:
        raise ValueError("❌ Loaded file is empty.")

    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(how="all")

    # Optional: Convert "Date" to Month
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.to_period("M").astype(str)

    return df

def find_best_column(df, expected):
    possibilities = COLUMN_SYNONYMS.get(expected, [expected])
    for term in possibilities:
        match = difflib.get_close_matches(term.lower(), [c.lower() for c in df.columns], n=1, cutoff=0.4)
        if match:
            for col in df.columns:
                if col.lower() == match[0]:
                    if expected.lower() == "revenue" and not pd.api.types.is_numeric_dtype(df[col]):
                        continue
                    return col
    return None

def infer_column_roles(df, api_key):
    genai.configure(api_key=api_key)

    columns = ", ".join(df.columns)
    try:
        sample = df.head(10).to_markdown(index=False)
    except Exception:
        sample = df.head(10).to_string(index=False)

    prompt = f"""
You are a business analyst. Based on the sample data below, determine which columns most likely represent:

- Revenue (or total income)
- Product (or item name)
- Region (or sales location)
- Month or date

Respond in this JSON format:
{{
  "Revenue": "<column_name>",
  "Product": "<column_name>",
  "Region": "<column_name>",
  "Month": "<column_name>"
}}

Columns: {columns}

Sample Data:
{sample}
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    import json
    try:
        inferred = json.loads(response.text.strip())
        if not isinstance(inferred, dict) or not all(k in inferred for k in ["Revenue", "Product", "Region", "Month"]):
            raise ValueError("Gemini did not return a complete mapping.")
    except Exception:
        inferred = {}

    return inferred

