# module/data_utils.py
import os
import json
import pandas as pd
import chardet
import difflib
import google.generativeai as genai

def load_and_clean_data(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".csv":
        with open(filepath, "rb") as f:
            encoding = chardet.detect(f.read())["encoding"]
        df = pd.read_csv(filepath, encoding=encoding)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        raise ValueError("Unsupported file type.")
    df.columns = df.columns.str.strip()
    return df

def infer_column_roles(df, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    schema = df.head(10).to_markdown(index=False)
    prompt = f"""
You are a data analyst. Infer the following roles from the table:
- Revenue (numeric sales or money)
- Product (item/category)
- Region (location or geography)
- Month (date, time, or period)

Return a dictionary like:
{{"Revenue": "Column1", "Product": "Column2", "Region": "Column3", "Month": "Column4"}}

Use ONLY columns found in this list:
{list(df.columns)}

Table Sample:
{schema}
"""
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text.strip())
    except Exception:
        return {}

def normalize_column_name(name, columns):
    if not name:
        return None
    match = difflib.get_close_matches(name, list(columns), n=1, cutoff=0.6)
    return match[0] if match else None

def find_fallback_column(df, roles, role_key, numeric=False):
    """Fallback if Gemini returns a column that doesn't exist"""
    # Try role-mapped column first
    if role_key in roles:
        candidate = normalize_column_name(roles[role_key], df.columns)
        if candidate:
            return candidate
    # Fallback: first numeric or first available
    if numeric:
        for col in df.select_dtypes("number").columns:
            return col
    else:
        return df.columns[0]
