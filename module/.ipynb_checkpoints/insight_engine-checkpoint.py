import google.generativeai as genai

def generate_insight_from_df(df, prompt_template):
    preview = df.head(20).to_markdown(index=False)
    prompt = prompt_template  # Already formatted in app.py
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()