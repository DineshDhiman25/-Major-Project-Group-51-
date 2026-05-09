"""
data/annotate_no.py
Converts a CSV of HRV-negative annotations into an Alpaca-style JSONL dataset.

Shared constants are imported from annotate_yes to avoid duplication.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Import shared constants from sibling module
sys.path.insert(0, str(Path(__file__).parent))
from annotate_yes import INSTRUCTION_TEXT  # noqa: E402

NO_RESPONSE = (
    "No, this post does not describe any human rights violation. "
    "It does not contain harm to civilians, destruction of civilian property, "
    "sexual violence, torture, execution, or abuse of detainees. "
    "The content appears neutral and does not meet the criteria for a human rights violation."
)


def build_no_jsonl(input_csv: str, output_path: str) -> None:
    """
    Convert *input_csv* (HRV-negative annotations) into an Alpaca-style JSONL
    fine-tuning dataset, writing both original-language and English-translation
    entries.

    Parameters
    ----------
    input_csv   : CSV with columns [Post, Trans].
    output_path : Destination .jsonl file.
    """
    df = pd.read_csv(input_csv)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            for post_text in (row["Post"], row["Trans"]):
                f.write(json.dumps({
                    "instruction": INSTRUCTION_TEXT,
                    "input": post_text,
                    "output": NO_RESPONSE,
                }, ensure_ascii=False) + "\n")

    print(f"✅ NO fine-tune JSONL saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build NO fine-tune JSONL")
    parser.add_argument("--input",  required=True, help="HRV-negative CSV path")
    parser.add_argument("--output", required=True, help="Output .jsonl path")
    args = parser.parse_args()

    build_no_jsonl(input_csv=args.input, output_path=args.output)
