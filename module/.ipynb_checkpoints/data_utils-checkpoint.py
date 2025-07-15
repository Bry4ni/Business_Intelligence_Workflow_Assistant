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
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(inplace=True)
    df['Month'] = df['Date'].dt.to_period('M')
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

