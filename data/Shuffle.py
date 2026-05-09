"""
data/Shuffle.py
Shuffles JSONL files row-by-row for reproducible randomisation.
"""

from __future__ import annotations

import json
import random


def shuffle_jsonl(input_path: str, output_path: str, seed: int = 42) -> None:
    """
    Read a JSONL file, shuffle its rows, and write to *output_path*.

    Parameters
    ----------
    input_path  : Source .jsonl file.
    output_path : Destination .jsonl file (may be same as input for in-place shuffle).
    seed        : Random seed for reproducibility.
    """
    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    with open(output_path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Shuffled {len(data)} entries → {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shuffle JSONL files")
    parser.add_argument("--files", nargs="+", required=True, help="JSONL files to shuffle in-place")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    for path in args.files:
        shuffle_jsonl(input_path=path, output_path=path, seed=args.seed)
