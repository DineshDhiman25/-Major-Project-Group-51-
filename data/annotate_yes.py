"""
data/annotate_yes.py
Converts a CSV of HRV-positive annotations into an Alpaca-style JSONL dataset.

Shared constants (INSTRUCTION_TEXT, TYPE_MAP) are defined here so that
annotate_no.py can import them without duplication.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Shared constants (also imported by annotate_no.py)
# ---------------------------------------------------------------------------

INSTRUCTION_TEXT = (
    "Analyze the post and determine whether it describes a human rights violation. "
    "If it does, explain the type in detail. If not, clearly state that no violation is present."
)

# Maps single or combined violation type keys → detailed explanations
TYPE_MAP: dict[str, str] = {
    "A": (
        "Yes, this post describes a human rights violation where military or government forces "
        "intentionally harm or kill civilians. It may involve direct attacks on unarmed "
        "individuals, deliberate use of lethal force, or actions that result in civilian "
        "casualties. The core focus is violence directed at non-combatants."
    ),
    "B": (
        "Yes, this post describes a human rights violation involving deliberate destruction of "
        "civilian property or infrastructure. It includes attacks on homes, residential "
        "buildings, or essential facilities such as hospitals, schools, and utilities. The "
        "emphasis is on targeted damage meant to intimidate, displace, or harm civilian life."
    ),
    "C": (
        "Yes, this post describes a human rights violation involving sexual violence, severe "
        "torture, or execution. It includes acts intended to cause extreme suffering, coercion, "
        "fear, or death. The emphasis is on serious bodily harm or coercive, violent actions "
        "against individuals."
    ),
    "D": (
        "Yes, this post describes a human rights violation involving mistreatment, coercion, "
        "intimidation, or abuse of detainees or prisoners. It may involve forced confessions, "
        "physical or psychological pressure, humiliation, or denial of basic rights. The focus "
        "is on abuse that occurs during detention, interrogation, or captivity."
    ),
    # --- Two-category combinations ---
    "A, B": (
        "Yes, this post describes multiple human rights violations, including both deliberate "
        "harm to civilians and intentional destruction of civilian property or infrastructure. "
        "Civilians may be attacked directly, and their homes or essential facilities may be "
        "damaged or destroyed. The post reflects a combination of violence against people and "
        "violence against civilian structures."
    ),
    "B, A": (
        "Yes, this post describes multiple human rights violations, including both deliberate "
        "harm to civilians and intentional destruction of civilian property or infrastructure. "
        "Civilians may be attacked directly, and their homes or essential facilities may be "
        "damaged or destroyed. The post reflects a combination of violence against people and "
        "violence against civilian structures."
    ),
    "A, C": (
        "Yes, this post describes multiple human rights violations including harm or killing of "
        "civilians and sexual violence, torture, or execution. It indicates that civilians are "
        "not only attacked directly but may also be subjected to severe abuse or extreme "
        "violence. This combination reflects high-intensity violence targeting civilian "
        "populations."
    ),
    "C, A": (
        "Yes, this post describes multiple human rights violations including harm or killing of "
        "civilians and sexual violence, torture, or execution. It indicates that civilians are "
        "not only attacked directly but may also be subjected to severe abuse or extreme "
        "violence. This combination reflects high-intensity violence targeting civilian "
        "populations."
    ),
    "A, D": (
        "Yes, this post describes human rights violations involving direct harm or killing of "
        "civilians and abuse or mistreatment of detainees. Civilians may be attacked outright, "
        "while those captured or detained may face coercion, intimidation, or degrading "
        "treatment. This combination reflects violence both in open environments and within "
        "detention settings."
    ),
    "D, A": (
        "Yes, this post describes human rights violations involving direct harm or killing of "
        "civilians and abuse or mistreatment of detainees. Civilians may be attacked outright, "
        "while those captured or detained may face coercion, intimidation, or degrading "
        "treatment. This combination reflects violence both in open environments and within "
        "detention settings."
    ),
    "C, D": (
        "Yes, this post describes multiple human rights violations involving sexual violence, "
        "torture, or execution, as well as mistreatment or coercive abuse of detainees. It "
        "reflects both extreme violence and abusive treatment of individuals held in custody "
        "or under control."
    ),
    "D, C": (
        "Yes, this post describes multiple human rights violations involving sexual violence, "
        "torture, or execution, as well as mistreatment or coercive abuse of detainees. It "
        "reflects both extreme violence and abusive treatment of individuals held in custody "
        "or under control."
    ),
    # --- Three-category combinations ---
    "A, C, D": (
        "Yes, this post describes several severe human rights violations, including harm or "
        "killing of civilians, sexual violence, torture, or execution, and mistreatment or "
        "coercive abuse of detainees. Civilians may be attacked directly, subjected to extreme "
        "violence, or abused while detained. This combination indicates broad, multi-layered "
        "violations against both civilians and detainees."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def expand_types(type_string: "str | float") -> "str | None":
    """Map a raw 'Type' cell (e.g. 'A, B') to its long-form explanation."""
    if pd.isna(type_string):
        return None
    type_string = str(type_string).strip()
    if type_string in TYPE_MAP:
        return TYPE_MAP[type_string]
    parts = [t.strip() for t in type_string.split(",")]
    sentences = [TYPE_MAP.get(p, f"Unknown type: {p}") for p in parts]
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_yes_jsonl(input_csv: str, output_path: str) -> None:
    """
    Convert *input_csv* (HRV-positive annotations) into an Alpaca-style JSONL
    fine-tuning dataset, writing both original-language and English-translation
    entries.

    Parameters
    ----------
    input_csv   : CSV with columns [Post, Trans, Type].
    output_path : Destination .jsonl file.
    """
    df = pd.read_csv(input_csv)
    df["Type_Expanded"] = df["Type"].apply(expand_types)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            explanation = row["Type_Expanded"]
            for post_text in (row["Post"], row["Trans"]):
                f.write(json.dumps({
                    "instruction": INSTRUCTION_TEXT,
                    "input": post_text,
                    "output": explanation,
                }, ensure_ascii=False) + "\n")

    print(f"✅ YES fine-tune JSONL saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build YES fine-tune JSONL")
    parser.add_argument("--input",  required=True, help="HRV-positive CSV path")
    parser.add_argument("--output", required=True, help="Output .jsonl path")
    args = parser.parse_args()

    build_yes_jsonl(input_csv=args.input, output_path=args.output)
