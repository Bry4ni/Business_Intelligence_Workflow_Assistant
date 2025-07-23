import os
import pandas as pd
import chardet
import google.generativeai as genai

# Configure Gemini once
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def detect_encoding(filepath):
    """Detect file encoding to safely load CSVs."""
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

def load_and_clean_data(filepath):
    """Load CSV or Excel with automatic detection."""
    ext = os.path.splitext(filepath)[-1].lower()

    try:
        if ext == ".csv":
            encoding = detect_encoding(filepath)
            df = pd.read_csv(filepath, encoding=encoding)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        raise ValueError(f"❌ File read error: {e}")

    # Drop all-null columns/rows
    df.dropna(how='all', axis=0, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)

    return df

def infer_column_roles(df, api_key):
    """Use Gemini to infer common column roles like Revenue, Product, etc."""
    genai.configure(api_key=api_key)

    prompt = f"""
You are a data scientist. Analyze the column names below and infer their roles.
Return a JSON object like this (use the closest matches available):

{{
  "Revenue": "RevenueColumnName",
  "Product": "ProductColumnName",
  "Region": "RegionColumnName",
  "Month": "MonthColumnName"
}}

Columns:
{list(df.columns)}
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()

        import json
        roles = json.loads(text)
        return roles
    except Exception as e:
        raise ValueError(f"❌ Column inference failed: {e}")
