# module/data_utils.py
import pandas as pd
import chardet
import os
import google.generativeai as genai
from difflib import get_close_matches

def load_and_clean_data(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".csv":
        with open(filepath, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
        return pd.read_csv(filepath, encoding=encoding)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(filepath, engine="openpyxl")
    else:
        raise ValueError("Unsupported file type.")

def infer_column_roles(df, api_key):
    genai.configure(api_key=api_key)
    sample = df.head(10).to_markdown(index=False)
    column_list = ", ".join(df.columns)
    prompt = f"""
You are a smart data scientist. Identify the role of each column from this list:
{column_list}

Sample:
{sample}

Return a dictionary like:
{{"Revenue": "Revenue", "Product": "Product", "Region": "Region", "Month": "Month"}}
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    try:
        return eval(response.text.strip())
    except:
        return {}

def normalize_column_name(requested, actual_columns):
    if not requested:
        return None
    matches = get_close_matches(requested, actual_columns, n=1, cutoff=0.6)
    return matches[0] if matches else requested
