#Type "libretranslate" in terminal to start, then run the code

import pandas as pd
import requests
import time

# === Settings ===
csv_path = "annotated/sentences/NHRV/posts_2.csv"
save_path = "annotated/sentences/NHRV/posts_2_t.csv"
api_url = "http://localhost:5000/translate"

# === Load Data ===
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows")

# Ensure 'Trans' column exists
if "Trans" not in df.columns:
    df["Trans"] = ""

# === Translate each post ===
for i, row in df.iterrows():
    text = str(row["Post"]).strip()
    if not text or text.lower() == "nan":
        continue

    # Skip if already translated
    if pd.notna(row["Trans"]) and str(row["Trans"]).strip():
        continue

    try:
        payload = {
            "q": text,
            "source": "auto",   # Auto-detect Russian/Ukrainian
            "target": "en",     # Translate to English
            "format": "text"
        }

        response = requests.post(api_url, data=payload, timeout=30)
        response.raise_for_status()

        translated = response.json()["translatedText"]

        # Update the DataFrame
        df.at[i, "Trans"] = translated

        print(f"[{i}] Translated: {text[:40]}... → {translated[:40]}")

        # Save progress every 20 translations
        if i % 20 == 0:
            df.to_csv(save_path, index=False)

        # Be kind to the local server
        time.sleep(0.3)

    except Exception as e:
        print(f"[{i}] ⚠️ Error: {e}")
        time.sleep(2)

# === Final Save ===
df.to_csv(save_path, index=False)
print(f"\nTranslation completed. Saved to: {save_path}")
