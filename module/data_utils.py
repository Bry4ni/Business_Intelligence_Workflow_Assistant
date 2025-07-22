# module/data_utils.py

import os
import pandas as pd
import difflib
import chardet
import google.generativeai as genai

COLUMN_SYNONYMS = {
    "Revenue": ["Revenue", "Sales", "Income", "Total", "Amount", "Ingresos"],
    "Product": ["Product", "Item", "Name", "Producto"],
    "Region": ["Region", "Area", "Zone", "Territory", "Región"],
    "Month": ["Month", "Date", "Periodo", "Mes"]
}

def load_and_clean_data(filepath):
    ext = os.path.splitext(filepath)[-1].lower()

    if ext in [".xls", ".xlsx"]:
        try:
            import openpyxl
            df = pd.read_excel(filepath, engine="openpyxl")
        except Exception as e:
            raise ValueError(f"❌ Excel read error: {e}")
    elif ext in [".csv", ".txt"]:
        with open(filepath, "rb") as f:
            raw = f.read(10000)
            result = chardet.detect(raw)
            encoding = result["encoding"] or "utf-8"

        try:
            df = pd.read_csv(filepath, encoding=encoding)
        except Exception as e:
            raise ValueError(f"❌ CSV read error with encoding {encoding}: {e}")
    else:
        raise ValueError("❌ Unsupported file type. Please upload a .csv or .xlsx file.")

    if df.empty:
        raise ValueError("❌ Loaded file is empty.")

    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(how="all")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.to_period("M").astype(str)

    return df

def infer_column_roles(df, api_key):
    genai.configure(api_key=api_key)

    columns = ", ".join(df.columns)
    try:
        sample = df.head(10).to_markdown(index=False)
    except Exception:
        sample = df.head(10).to_string(index=False)

    prompt = f"""
You are a business analyst. Based on the sample data below, identify:

- Revenue column
- Product column
- Region column
- Month or Date column

Respond in JSON format:
{{
  "Revenue": "<col>",
  "Product": "<col>",
  "Region": "<col>",
  "Month": "<col>"
}}

Columns: {columns}
Sample:
{sample}
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    import json
    try:
        inferred = json.loads(response.text.strip())
    except Exception:
        inferred = {}

    # Fallback to fuzzy matching
    for role in ["Revenue", "Product", "Region", "Month"]:
        if role not in inferred or inferred[role] not in df.columns:
            inferred[role] = find_best_column(df, role)

    return inferred

def find_best_column(df, role):
    candidates = COLUMN_SYNONYMS.get(role, [role])
    for name in candidates:
        match = difflib.get_close_matches(name.lower(), [c.lower() for c in df.columns], n=1, cutoff=0.6)
        if match:
            for col in df.columns:
                if col.lower() == match[0]:
                    if role == "Revenue" and not pd.api.types.is_numeric_dtype(df[col]):
                        continue
                    return col
    return None

