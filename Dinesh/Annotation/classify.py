import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm
import time


INPUT_FILE = "translated_posts.csv"
OUTPUT_FILE = "annotated_posts.csv"
CHECKPOINT_INTERVAL = 50
SLEEP_BETWEEN_CALLS = 0.5
MODEL_NAME = "gemini-2.5-flash"
MAX_ROWS = 5000

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

df_full = pd.read_csv(INPUT_FILE)
df = df_full.head(MAX_ROWS).copy()
print(f"Loaded {len(df)} rows (limited to {MAX_ROWS})")

# Resume logic
if os.path.exists(OUTPUT_FILE):
    df_existing = pd.read_csv(OUTPUT_FILE)
    if "HRV" in df_existing.columns:
        processed = df_existing["HRV"].notna().sum()
        print(f"Resuming from row {processed}/{len(df)}")
        df = df_existing.head(MAX_ROWS)
    else:
        df["HRV"] = None
        processed = 0
else:
    df["HRV"] = None
    processed = 0


model = genai.GenerativeModel(MODEL_NAME)

def make_prompt(row):
    return f"""
You are an expert detecting human rights violations (HRV) in social media posts
about the Russia–Ukraine conflict.

A post should be marked **Yes** if it clearly describes or reports:
  • Killing of civilians
  • Destruction of civil objects (homes, hospitals, schools, markets)
  • Rape, torture, execution (war crimes)
  • Mistreatment of prisoners
Otherwise, mark **No**.

Return output in JSON as:
{{"HRV": "Yes" or "No"}}

---
Channel ID: {row['Channel ID']}
Post ID: {row['Post ID']}
Date: {row['Date']}
Original (Russian): {row['Post']}
English Translation: {row['post_translated']}
---
"""

for i in tqdm(range(processed, len(df))):
    try:
        row = df.iloc[i]
        prompt = make_prompt(row)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        text = response.text.strip()

        if '"Yes"' in text or 'Yes' in text:
            label = "Yes"
        elif '"No"' in text or 'No' in text:
            label = "No"
        else:
            label = "Unclear"

        df.at[i, "HRV"] = label

    except Exception as e:
        print(f"Error on row {i}: {e}")
        df.at[i, "HRV"] = "Error"

    if (i + 1) % CHECKPOINT_INTERVAL == 0:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Checkpoint saved at row {i + 1}")

    time.sleep(SLEEP_BETWEEN_CALLS)

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved final results to {OUTPUT_FILE}")
