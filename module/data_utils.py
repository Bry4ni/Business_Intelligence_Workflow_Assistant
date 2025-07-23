import os
import chardet
import pandas as pd
import google.generativeai as genai

# Load file and clean data
def load_and_clean_data(filepath):
    ext = os.path.splitext(filepath)[-1].lower()

    if ext == ".csv":
        with open(filepath, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result["encoding"]

        try:
            df = pd.read_csv(filepath, encoding=encoding)
        except Exception as e:
            raise ValueError(f"❌ CSV read error: {e}")

    elif ext in [".xls", ".xlsx"]:
        try:
            df = pd.read_excel(filepath, engine="openpyxl")
        except Exception as e:
            raise ValueError(f"❌ Excel read error: {e}")
    else:
        raise ValueError("❌ Unsupported file type. Please upload a .csv or .xlsx file.")

    df.columns = df.columns.str.strip()  # Clean column names
    return df

# Infer column roles using Gemini
def infer_column_roles(df, api_key, model_name="gemini-2.0-pro"):
    genai.configure(api_key=api_key)

    # Prepare sample and column names
    sample = df.head(10).to_markdown(index=False)
    columns = ", ".join(df.columns)

    prompt = f"""
You are a data expert. Your task is to infer column roles for the following dataset sample.

Return a JSON object mapping roles: Revenue, Product, Region, Month.
Respond ONLY with a JSON object using actual column names from the sample data.

Example:
{{"Revenue": "Revenue Column", "Product": "Product Column", "Region": "Region Column", "Month": "Month Column"}}

Columns: {columns}
Sample:
{sample}
"""

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    text = response.text.strip()

    import json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
