from pathlib import Path

def load_insights(insight_path):
    try:
        return Path(insight_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return "*No insights found. Please generate insights using the Gemini API and save to `insights.txt`.*"

def ensure_report_folder(path="../reports"):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path