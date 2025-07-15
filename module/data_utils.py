# Directory: module/data_utils.py

import pandas as pd
import os
import difflib
import google.generativeai as genai

COLUMN_SYNONYMS = {
    "Revenue": ["Revenue", "Sales", "Total", "Income", "Ingresos", "Amount"],
    "Product": ["Product", "Item", "Producto", "Product_Name"],
    "Region": ["Region", "Area", "Territory", "Zone", "Región"],
    "Month": ["Month", "Mes", "Periodo", "Period"]
}

def load_and_clean_data(filepath):
    ext = os.path.splitext(filepath)[-1].lower()

    # Try Excel first if file extension is correct
    if ext in [".xls", ".xlsx"]:
        try:
            df = pd.read_excel(filepath)
        except Exception as e:
            raise ValueError(f"❌ Failed to read Excel file: {str(e)}")
    else:
        # Try multiple encodings for CSV
        for encoding in ["utf-8", "utf-8-sig", "ISO-8859-1", "cp1252"]:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("❌ Failed to decode CSV file. Try saving it as UTF-8 or upload as Excel.")

    # Clean up column names
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(how="all")

    # Add 'Month' column if 'Date' exists
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
                    # ⬇️ NEW: Require numeric if expected is Revenue
                    if expected.lower() == "revenue":
                        if pd.api.types.is_numeric_dtype(df[col]):
                            return col
                    else:
                        return col
    return None

def infer_column_roles(df, api_key):
    genai.configure(api_key=api_key)

    columns = ", ".join(df.columns)
    sample = df.head(10).to_markdown(index=False)

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
    except Exception:
        inferred = {}

    return inferred

