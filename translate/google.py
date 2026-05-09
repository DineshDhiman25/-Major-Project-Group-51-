import pandas as pd
import time
from google.cloud import translate_v2 as translate

df = pd.read_csv("/Major_Grp51/Aditya/dataset/Venezuela/hrv_train1.csv")

translate_client = translate.Client()

TARGET_LANG = "en"
SOURCE_LANG = "es_MX"

df["Trans"] = None

for count, (idx, row) in enumerate(df["text_preview"].items(), start=1):
    try:
        result = translate_client.translate(
            values=str(row),
            target_language=TARGET_LANG,
            source_language=SOURCE_LANG,
        )
        df.at[idx, "Trans"] = result["translatedText"]
    except Exception as e:
        print(f"[Row {idx}] Error: {e}")
        df.at[idx, "Trans"] = None

    if count % 10 == 0:
        print(f"Translated {count}/{len(df)}")
        df.to_csv("/Major_Grp51/Aditya/dataset/Venezuela/hrv_train1_translated.csv", index=False)

# Final save
df.to_csv("/Major_Grp51/Aditya/dataset/Venezuela/hrv_train_translated.csv", index=False)
print("Done.")
