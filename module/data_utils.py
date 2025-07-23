# module/data_utils.py
import json
import os
import pandas as pd
import chardet
import difflib
import google.generativeai as genai

# Load and clean uploaded data
def load_and_clean_data(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    try:
        if ext == ".csv":
            with open(filepath, "rb") as f:
                encoding = chardet.detect(f.read())["encoding"]
            df = pd.read_csv(filepath, encoding=encoding)
        elif ext in [".xls", ".xlsx"]:
            import openpyxl  # Ensure dependency
            df = pd.read_excel(filepath, engine="openpyxl")
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        raise ValueError(f"❌ File read error: {e}")
    
    df.columns = df.columns.str.strip()  # Strip whitespace from column headers
    return df

# Use Gemini to infer column roles (Revenue, Product, etc.)
def infer_column_roles(df, api_key):
    genai.configure(api_key=api_key)  # Only configure once
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    schema = df.head(10).to_markdown(index=False)

    prompt = f"""
You are a data analyst. Infer the following roles based on the table below:
- Revenue (numeric sales values)
- Product (categories or item types)
- Region (geographic labels)
- Month (time period or date grouping)

Respond ONLY in JSON format like:
{{
  "Revenue": "ColumnA",
  "Product": "ColumnB",
  "Region": "ColumnC",
  "Month": "ColumnD"
}}

Table:
{schema}
"""
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text.strip())
    except Exception as e:
        print("❌ Gemini response parse error:", e)
        print("🔁 Raw response:", response.text)
        return {}

# Normalize column names using fuzzy matching
def normalize_column_name(name, columns):
    if not name:
        return None
    columns = list(columns)
    match = difflib.get_close_matches(name, columns, n=1, cutoff=0.6)
    if match:
        return match[0]
    return name  # Return as-is if no good match found

