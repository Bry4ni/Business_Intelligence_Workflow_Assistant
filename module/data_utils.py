import os
import pandas as pd
import chardet
import google.generativeai as genai

# Load and clean CSV or Excel file
def load_and_clean_data(filepath):
    try:
        if filepath.endswith('.csv'):
            with open(filepath, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            df = pd.read_csv(filepath, encoding=encoding)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath, engine='openpyxl')
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        raise ValueError(f"❌ File read error: {e}")

    # Drop empty columns and rows
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)

    # Normalize column names
    df.columns = [col.strip().title() for col in df.columns]

    return df


# Gemini-powered column role inference
def infer_column_roles(df, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
You are a data analyst AI. The user will give you a dataset with column names.
Your job is to identify the most likely role for each column. Use this list of roles:

- Revenue
- Product
- Region
- Month
- Date
- Units Sold
- Unit Price
- Customer
- Category

Return a Python dictionary that maps actual column names to inferred roles.

Here are the column names:
{list(df.columns)}
"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Try to parse dictionary from response
        roles = eval(response_text) if "{" in response_text else {}
        return roles if isinstance(roles, dict) else {}
    except Exception as e:
        raise RuntimeError(f"❌ Failed to infer column roles: {e}")
