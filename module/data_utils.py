import pandas as pd
import os
import difflib
import chardet
import google.generativeai as genai

# Column role synonyms
COLUMN_SYNONYMS = {
    "Revenue": ["Revenue", "Sales", "Total", "Income", "Ingresos", "Amount", "Ventas"],
    "Product": ["Product", "Item", "Producto", "Product_Name", "商品"],
    "Region": ["Region", "Area", "Territory", "Zone", "Región", "地区"],
    "Month": ["Month", "Mes", "Periodo", "Period", "Date", "Fecha", "月份"]
}

# ------------------------------------------
# Load and clean uploaded CSV or Excel file
# ------------------------------------------
def load_and_clean_data(filepath):
    ext = os.path.splitext(filepath)[-1].lower()

    if ext in [".xls", ".xlsx"]:
        try:
            df = pd.read_excel(filepath)
        except Exception as e:
            raise ValueError(f"❌ Excel read error: {e}")
    elif ext in [".csv", ".txt"]:
        with open(filepath, "rb") as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result["encoding"] or "utf-8"

        try:
            df = pd.read_csv(filepath, encoding=encoding)
        except Exception as e:
            raise ValueError(f"❌ CSV read error with encoding {encoding}: {e}")
    else:
        raise ValueError("❌ Unsupported file type. Please upload a .csv or .xlsx file.")

    if df.empty:
        raise ValueError("❌ Loaded file is empty.")

    # Clean headers and remove all-empty rows
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(how="all")

    # Convert 'Date' column if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.to_period("M").astype(str)

    # Standardize column names using synonyms
    df = substitute_column_names(df)

    return df

# ------------------------------------------
# Substitute synonyms with standardized names
# ------------------------------------------
def substitute_column_names(df):
    renamed = {}
    lower_cols = [col.lower() for col in df.columns]

    for standard_name, variants in COLUMN_SYNONYMS.items():
        for v in variants:
            if v.lower() in lower_cols:
                match_idx = lower_cols.index(v.lower())
                current_name = df.columns[match_idx]
                renamed[current_name] = standard_name
                break  # Stop after the first match

    return df.rename(columns=renamed)

# ------------------------------------------
# Fuzzy fallback if Gemini fails
# ------------------------------------------
def find_best_column(df, expected):
    possibilities = COLUMN_SYNONYMS.get(expected, [expected])
    for term in possibilities:
        match = difflib.get_close_matches(term.lower(), [c.lower() for c in df.columns], n=1, cutoff=0.4)
        if match:
            for col in df.columns:
                if col.lower() == match[0]:
                    if expected.lower() == "revenue":
                        if pd.api.types.is_numeric_dtype(df[col]):
                            return col
                    else:
                        return col
    return None

# ------------------------------------------
# Gemini-powered AI role inference
# ------------------------------------------
def infer_column_roles(df, api_key):
    genai.configure(api_key=api_key)

    columns = ", ".join(df.columns)
    try:
        sample = df.head(10).to_markdown(index=False)
    except Exception:
        sample = df.head(10).to_string(index=False)

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
        if not isinstance(inferred, dict) or not all(k in inferred for k in ["Revenue", "Product", "Region", "Month"]):
            raise ValueError("Gemini did not return a complete mapping.")
    except Exception:
        inferred = {}

    return inferred

