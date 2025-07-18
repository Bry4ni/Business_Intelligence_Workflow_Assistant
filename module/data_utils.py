import pandas as pd
import os
import difflib
import chardet
import google.generativeai as genai

COLUMN_SYNONYMS = {
    "Revenue": ["Revenue", "Sales", "Total", "Income", "Ingresos", "Amount"],
    "Product": ["Product", "Item", "Producto", "Product_Name"],
    "Region": ["Region", "Area", "Territory", "Zone", "Región"],
    "Month": ["Month", "Mes", "Periodo", "Period"]
}

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
            encoding = chardet.detect(raw_data)['encoding'] or "utf-8"

        try:
            df = pd.read_csv(filepath, encoding=encoding)
        except Exception as e:
            raise ValueError(f"❌ CSV read error with encoding {encoding}: {e}")
    else:
        raise ValueError("❌ Unsupported file type. Please upload a .csv or .xlsx file.")

    if df.empty:
        raise ValueError("❌ Loaded file is empty.")

    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(how="all")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.to_period("M").astype(str)

    return df
