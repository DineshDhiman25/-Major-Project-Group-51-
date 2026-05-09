import pandas as pd
import json

type_map = {
    "A": "Yes, this post describes a human rights violation where military or government forces intentionally harm or kill civilians. It may involve direct attacks on unarmed individuals, deliberate use of lethal force, or actions that result in civilian casualties. The core focus is violence directed at non-combatants.",

    "B": "Yes, this post describes a human rights violation involving deliberate destruction of civilian property or infrastructure. It includes attacks on homes, residential buildings, or essential facilities such as hospitals, schools, and utilities. The emphasis is on targeted damage meant to intimidate, displace, or harm civilian life.",

    "C": "Yes, this post describes a human rights violation involving sexual violence, severe torture, or execution. It includes acts intended to cause extreme suffering, coercion, fear, or death. The emphasis is on serious bodily harm or coercive, violent actions against individuals.",

    "D": "Yes, this post describes a human rights violation involving mistreatment, coercion, intimidation, or abuse of detainees or prisoners. It may involve forced confessions, physical or psychological pressure, humiliation, or denial of basic rights. The focus is on abuse that occurs during detention, interrogation, or captivity.",

    # --- Two-category combinations ---
    "A, B": "Yes, this post describes multiple human rights violations, including both deliberate harm to civilians and intentional destruction of civilian property or infrastructure. Civilians may be attacked directly, and their homes or essential facilities may be damaged or destroyed. The post reflects a combination of violence against people and violence against civilian structures.",
    "B, A": "Yes, this post describes multiple human rights violations, including both deliberate harm to civilians and intentional destruction of civilian property or infrastructure. Civilians may be attacked directly, and their homes or essential facilities may be damaged or destroyed. The post reflects a combination of violence against people and violence against civilian structures.",

    "A, C": "Yes, this post describes multiple human rights violations including harm or killing of civilians and sexual violence, torture, or execution. It indicates that civilians are not only attacked directly but may also be subjected to severe abuse or extreme violence. This combination reflects high-intensity violence targeting civilian populations.",
    "C, A": "Yes, this post describes multiple human rights violations including harm or killing of civilians and sexual violence, torture, or execution. It indicates that civilians are not only attacked directly but may also be subjected to severe abuse or extreme violence. This combination reflects high-intensity violence targeting civilian populations.",

    "A, D": "Yes, this post describes human rights violations involving direct harm or killing of civilians and abuse or mistreatment of detainees. Civilians may be attacked outright, while those captured or detained may face coercion, intimidation, or degrading treatment. This combination reflects violence both in open environments and within detention settings.",
    "D, A": "Yes, this post describes human rights violations involving direct harm or killing of civilians and abuse or mistreatment of detainees. Civilians may be attacked outright, while those captured or detained may face coercion, intimidation, or degrading treatment. This combination reflects violence both in open environments and within detention settings.",

    "C, D": "Yes, this post describes multiple human rights violations involving sexual violence, torture, or execution, as well as mistreatment or coercive abuse of detainees. It reflects both extreme violence and abusive treatment of individuals held in custody or under control.",
    "D, C": "Yes, this post describes multiple human rights violations involving sexual violence, torture, or execution, as well as mistreatment or coercive abuse of detainees. It reflects both extreme violence and abusive treatment of individuals held in custody or under control.",

    # --- Three-category combinations ---

    "A, C, D": "Yes, this post describes several severe human rights violations, including harm or killing of civilians, sexual violence, torture, or execution, and mistreatment or coercive abuse of detainees. Civilians may be attacked directly, subjected to extreme violence, or abused while detained. This combination indicates broad, multi-layered violations against both civilians and detainees."
}

instruction_text = (
    "Analyze the post and determine whether it describes a human rights violation. "
    "If it does, explain the type in detail. If not, clearly state that no violation is present."
)

df_yes = pd.read_csv("Aditya/Annotation/hrv_yes.csv")

def expand_types(type_string):
    if pd.isna(type_string):
        return None

    type_string = type_string.strip()

    if type_string in type_map:
        return type_map[type_string]

    parts = [t.strip() for t in type_string.split(",")]
    sentences = [type_map.get(p, f"Unknown type: {p}") for p in parts]

    return " ".join(sentences)

df_yes["Type_Expanded"] = df_yes["Type"].apply(expand_types)

output_path = "hrv_finetune_yes.jsonl"
with open(output_path, "w", encoding="utf-8") as f:

    # YES SAMPLES (all)
    for _, row in df_yes.iterrows():
        post_ru = row["Post"]
        post_en = row["Trans"]
        explanation = row["Type_Expanded"]

        f.write(json.dumps({
            "instruction": instruction_text,
            "input": post_ru,
            "output": explanation
        }, ensure_ascii=False) + "\n")

        f.write(json.dumps({
            "instruction": instruction_text,
            "input": post_en,
            "output": explanation
        }, ensure_ascii=False) + "\n")
