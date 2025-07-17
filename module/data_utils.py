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
            return pd.read_excel(filepath)
        except Exception as e:
            raise ValueError(f"❌ Excel read error: {e}")
    else:
        # Try detecting encoding using chardet
        with open(filepath, "rb") as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result.get("encoding") or "utf-8"

        # Attempt to read using detected encoding
        try:
            df = pd.read_csv(filepath, encoding=encoding)
        except Exception as e1:
            # Retry with fallbacks
            for fallback in ["ISO-8859-1", "latin1", "cp1252"]:
                try:
                    df = pd.read_csv(filepath, encoding=fallback)
                    break
                except Exception:
                    df = None
            if df is None:
                raise ValueError(f"❌ CSV read error: {e1}")

    if df.empty:
        raise ValueError("❌ Loaded file is empty.")

    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(how="all")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.to_period("M").astype(str)

    return df

