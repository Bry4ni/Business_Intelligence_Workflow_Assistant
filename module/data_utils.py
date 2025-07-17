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
    else:
        # ✅ Auto-detect encoding for CSV files
        with open(filepath, "rb") as f:
            raw_data = f.read(10000)
            detected = chardet.detect(raw_data)
            encoding = detected["encoding"] or "utf-8"

        try:
            df = pd.read_csv(filepath, encoding=encoding)
        except Exception as e:
            # 🔁 Fallback encodings
            for fallback in ["ISO-8859-1", "cp1252", "latin1"]:
                try:
                    df = pd.read_csv(filepath, encoding=fallback)
                    break
                except Exception:
                    df = None
            if df is None:
                raise ValueError(f"❌ CSV read error: {e}")
    if df.empty:
        raise ValueError("❌ Loaded file is empty.")

    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(how="all")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.to_period("M").astype(str)

    return df

