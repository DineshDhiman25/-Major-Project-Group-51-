import pandas as pd
import json

df_no = pd.read_csv("Aditya/Annotation/hrv_no.csv")

instruction_text = (
    "Analyze the post and determine whether it describes a human rights violation. "
    "If it does, explain the type in detail. If not, clearly state that no violation is present."
)

no_response = (
    "No, this post does not describe any human rights violation. "
    "It does not contain harm to civilians, destruction of civilian property, "
    "sexual violence, torture, execution, or abuse of detainees. "
    "The content appears neutral and does not meet the criteria for a human rights violation."
)

output_path = "hrv_finetune_alpaca.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    # NO SAMPLES (sampled)
    for _, row in df_no.iterrows():
        post_ru = row["Post"]
        post_en = row["Trans"]

        f.write(json.dumps({
            "instruction": instruction_text,
            "input": post_ru,
            "output": no_response
        }, ensure_ascii=False) + "\n")

        f.write(json.dumps({
            "instruction": instruction_text,
            "input": post_en,
            "output": no_response
        }, ensure_ascii=False) + "\n")

print(f"Saved: {output_path}")
