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
    df = None

    try:
        if ext in [".xls", ".xlsx"]:
            df = pd.read_excel(filepath)
        elif ext == ".csv":
            # Auto-detect CSV encoding with chardet
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
    except Exception as e:
        raise ValueError(f"❌ File read error: {e}")

    if df is None or df.empty:
        raise ValueError("❌ Loaded file is empty or could not be read.")

    # Clean headers
    df.columns = [str(col).strip() for col in df.columns]

    # Drop fully empty rows
    df.dropna(how="all", inplace=True)

    # Optional: extract Month from a 'Date' column if available
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Month"] = df["Date"].dt.to_period("M").astype(str)
        except Exception:
            pass  # Leave as-is if conversion fails

    return df
