import json
import random

def shuffle_jsonl_file(input_path, output_path, seed=42):
    """
    Reads a JSONL file, shuffles rows, and saves to a new JSONL file.
    """
    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse each JSON line
    data = [json.loads(line) for line in lines]

    # Shuffle
    random.shuffle(data)

    # Write back
    with open(output_path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Shuffled file saved to: {output_path}")


# Shuffle train and test
shuffle_jsonl_file("hrv_train.jsonl", "hrv_train.jsonl")
shuffle_jsonl_file("hrv_test.jsonl", "hrv_test.jsonl")
shuffle_jsonl_file("hrv_val.jsonl", "hrv_val.jsonl")
