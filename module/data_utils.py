import os
import json
import pandas as pd
import chardet
import difflib
import google.generativeai as genai

# Load and clean uploaded file
def load_and_clean_data(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    try:
        if ext == ".csv":
            with open(filepath, "rb") as f:
                encoding = chardet.detect(f.read())["encoding"]
            df = pd.read_csv(filepath, encoding=encoding)
        elif ext in [".xls", ".xlsx"]:
            import openpyxl  # Ensure Excel support
            df = pd.read_excel(filepath, engine="openpyxl")
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        raise ValueError(f"❌ File read error: {e}")

    df.columns = df.columns.str.strip()  # Remove whitespace from headers
    return df

# Use Gemini to infer roles (Revenue, Product, Region, Month)
def infer_column_roles(df, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    table_sample = df.head(10).to_markdown(index=False)
    prompt = f"""
You are a multilingual data analyst.

From the table below, infer which column best represents each role:
- Revenue (numeric sales amount)
- Product (product names or types)
- Region (geographic area)
- Month (date or month field)

Respond ONLY in this JSON format:
{{
  "Revenue": "...",
  "Product": "...",
  "Region": "...",
  "Month": "..."
}}

Table:
{table_sample}
"""
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        return json.loads(raw)
    except Exception as e:
        print("⚠️ Failed to parse column roles:", e)
        return {}

# Normalize column names using fuzzy matching
def normalize_column_name(name, column_list):
    if not name:
        return None
    matches = difflib.get_close_matches(name, list(column_list), n=1, cutoff=0.6)
    return matches[0] if matches else name

# Clean Gemini's JSON response and fallback if needed
def clean_gemini_json(response_text):
    try:
        # Remove markdown formatting if accidentally returned
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        print("❌ Failed to clean/parse Gemini response:", e)
        return {
            "summary": "Gemini returned an unparseable response.",
            "charts": []
        }
