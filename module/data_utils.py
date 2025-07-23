import os
import pandas as pd
import chardet
import google.generativeai as genai

def load_and_clean_data(filepath):
    """
    Load a CSV or Excel file based on its extension.
    Uses chardet to detect encoding for CSV files.
    """
    ext = os.path.splitext(filepath)[-1].lower()
    try:
        if ext == '.csv':
            with open(filepath, 'rb') as f:
                result = chardet.detect(f.read())
                encoding = result['encoding']
            df = pd.read_csv(filepath, encoding=encoding)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath, engine="openpyxl")
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        return df
    except Exception as e:
        raise ValueError(f"❌ Excel read error: {e}")

def infer_column_roles(df, api_key=None):
    """
    Use Gemini to infer column roles (e.g. Revenue, Product, Region, Month) from the DataFrame.
    Returns a dictionary mapping standardized roles to actual column names.
    """
    if not api_key:
        raise ValueError("Gemini API key is missing.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    columns = list(df.columns)
    sample = df.head(10).to_dict(orient="records")

    prompt = f"""
You are a data expert. Identify which of the following DataFrame columns correspond to these standardized business roles:

- 'Revenue': the main numerical metric related to earnings or sales.
- 'Product': the item being sold or tracked.
- 'Region': geographic location, like country, state, area.
- 'Month': a date or time column that represents monthly data.

Respond in JSON format like:
{{
  "Revenue": "revenue_column",
  "Product": "product_column",
  "Region": "region_column",
  "Month": "month_column"
}}

Columns: {columns}
Sample data: {sample}
Only return the JSON dictionary.
"""

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        return eval(result) if result.startswith("{") else {}
    except Exception as e:
        raise ValueError(f"❌ Role inference failed: {e}")

