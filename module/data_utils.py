# module/data_utils.py
import os
import json
import chardet
import pandas as pd
import difflib
import google.generativeai as genai
import re

def load_and_clean_data(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    try:
        if ext == ".csv":
            with open(filepath, "rb") as f:
                encoding = chardet.detect(f.read())["encoding"]
            df = pd.read_csv(filepath, encoding=encoding)
        elif ext in [".xls", ".xlsx"]:
            import openpyxl
            df = pd.read_excel(filepath, engine="openpyxl")
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        raise ValueError(f"❌ File read error: {e}")
    
    df.columns = df.columns.astype(str).str.strip()
    return df

def infer_column_roles(df, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    markdown_sample = df.head(10).to_markdown(index=False)
    prompt = f"""
You are a data analyst. Identify the role of each column from this table:
Roles to identify: Revenue, Product, Region, Month.

Respond ONLY in JSON like:
{{
  "Revenue": "...",
  "Product": "...",
  "Region": "...",
  "Month": "..."
}}

Table:
{markdown_sample}
"""
    response = model.generate_content(prompt)
    try:
        cleaned = clean_gemini_json(response.text.strip())
        return json.loads(cleaned)
    except Exception as e:
        print("❌ Gemini role inference failed:", e)
        return {}

def normalize_column_name(name, columns):
    if not name:
        return None
    match = difflib.get_close_matches(name, columns, n=1, cutoff=0.6)
    return match[0] if match else name

def clean_gemini_json(text):
    # Remove Markdown code block formatting if present
    cleaned = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return cleaned
