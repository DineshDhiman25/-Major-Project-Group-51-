"""
Annotation/app.py
Gradio annotation interface for Human Rights Violation (HRV) labelling.
"""

from __future__ import annotations

import os

import pandas as pd


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

VIOLATION_TYPE_CHOICES = [
    "A: Killing or injury of civilians",
    "B: Destruction of civilian objects (homes, hospitals, schools, markets)",
    "C: Rape, torture, or execution",
    "D: Mistreatment or torture of prisoners/detainees",
]


def load_data(csv_file: str) -> tuple[int, pd.DataFrame]:
    """Load CSV, ensure required columns exist, and find the first unannotated row."""
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["Index", "Post", "Trans", "Label", "Type"])

    for col in ["Index", "Post", "Trans", "Label", "Type"]:
        if col not in df.columns:
            df[col] = None

    # Ensure string columns stay as object dtype even when all values are NaN.
    # Without this, pandas infers them as float64 and crashes when writing strings.
    for col in ["Label", "Trans", "Type"]:
        df[col] = df[col].astype(object)

    next_row_idx = len(df)
    for idx, row in df.iterrows():
        if pd.isna(row.get("Label", None)) or row.get("Label", "") == "":
            next_row_idx = idx
            break

    return next_row_idx, df


def save_data(df: pd.DataFrame, csv_file: str) -> str:
    os.makedirs(os.path.dirname(csv_file) or ".", exist_ok=True)
    df.to_csv(csv_file, index=False)
    return "Data saved successfully!"


def annotate_post(row_idx, label, violation_type, english_trans_text, df, csv_file: str):
    """Save current annotation and advance to the next unannotated row."""
    if row_idx < len(df):
        df.at[row_idx, "Label"] = label
        df.at[row_idx, "Trans"] = english_trans_text
        if isinstance(violation_type, list):
            formatted = ", ".join([t[0] for t in violation_type])
            df.at[row_idx, "Type"] = formatted
        else:
            df.at[row_idx, "Type"] = violation_type[0] if violation_type else ""

        save_data(df, csv_file)

        next_row = len(df)
        for idx, row in df.iterrows():
            if pd.isna(row.get("Label", None)) or row.get("Label", "") == "":
                next_row = idx
                break

        if next_row < len(df):
            current_row = df.iloc[next_row]
            post_display = f"Post {current_row.get('Index', next_row + 1)}"
            russian_post = current_row.get("Post", "")
            english_trans_text = current_row.get("Trans", "")
            status = "Next post loaded"
        else:
            post_display = "All posts annotated!"
            russian_post = ""
            english_trans_text = ""
            status = "All posts have been annotated. Great job!"

        return (next_row, df, post_display, russian_post, english_trans_text, status, None, None)

    return (row_idx, df, "No more posts to annotate", "", "", "Completed", None, None)


def reset_annotation(df, csv_file: str):
    row_idx, new_df = load_data(csv_file)
    if row_idx < len(new_df):
        current_row = new_df.iloc[row_idx]
        post_display = f"Post {current_row.get('Index', row_idx + 1)}"
        russian_post = current_row.get("Post", "")
        english_trans_text = current_row.get("Trans", "")
        status = "Reset to first unannotated post"
    else:
        post_display = "No posts to annotate"
        russian_post = ""
        english_trans_text = ""
        status = "No posts available"
    return (row_idx, new_df, post_display, russian_post, english_trans_text, status, None, None)


def preview_data(csv_file: str) -> str:
    if not os.path.exists(csv_file):
        return "No data file found."
    df = pd.read_csv(csv_file)
    total = len(df)
    annotated = len(df[df["Label"].notna() & (df["Label"] != "")])
    types_count = df["Type"].value_counts().to_dict() if "Type" in df.columns else {}
    stats = (
        f"**Dataset Statistics**\n"
        f"- Total Posts: {total}\n"
        f"- Annotated: {annotated}\n"
        f"- Remaining: {total - annotated}\n\n"
        f"**Type Distribution:**"
    )
    for type_key, count in types_count.items():
        if pd.notna(type_key) and type_key:
            stats += f"\n- {type_key}: {count}"
    return stats


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def run_gradio_app(
    csv_file: str = "Csv/hrv_train_venezuela.csv",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = True,
) -> None:
    """
    Launch the Gradio annotation interface.

    Parameters
    ----------
    csv_file    : Path to the CSV being annotated.
    server_port : Port for the local web server.
    share       : Whether to create a public Gradio share link.
    debug       : Enable Gradio debug mode.
    """
    import gradio as gr  # lazy import – optional dependency

    initial_row_idx, initial_df = load_data(csv_file)

    if initial_row_idx < len(initial_df):
        initial_row = initial_df.iloc[initial_row_idx]
        initial_post_display = f"Post {initial_row.get('Index', initial_row_idx + 1)}"
        initial_russian = initial_row.get("Post", "")
        initial_english = initial_row.get("Trans", "")
        initial_status = "Ready to annotate"
    else:
        initial_post_display = "No posts to annotate"
        initial_russian = ""
        initial_english = ""
        initial_status = "No data available"

    with gr.Blocks(title="Human Rights Violation Annotation Tool") as demo:
        gr.Markdown("# Human Rights Violation Annotation Tool")
        gr.Markdown(
            "Annotate social media posts for human rights violations.\n"
            "Destruction of Military facilities is **not** HRV."
        )

        row_idx_state = gr.State(value=initial_row_idx)
        df_state = gr.State(value=initial_df)

        with gr.Row():
            with gr.Column(scale=2):
                post_number = gr.Textbox(
                    value=initial_post_display, label="Current Post", interactive=False
                )
                russian_post = gr.Textbox(
                    value=initial_russian, label="Post (Original)", lines=4, interactive=False
                )
                english_trans = gr.Textbox(
                    value=initial_english, label="English Translation", lines=4, interactive=True
                )
                status = gr.Textbox(
                    value=initial_status, label="Status", interactive=False
                )

            with gr.Column(scale=1):
                gr.Markdown("### Annotation")
                label_choice = gr.Radio(
                    choices=["Yes", "No"],
                    label="Is this a Human Rights Violation?",
                    value=None,
                )
                type_choice = gr.CheckboxGroup(
                    choices=VIOLATION_TYPE_CHOICES, label="Violation Type", value=None
                )
                annotate_btn = gr.Button("Save Annotation & Next Post", variant="primary")
                reset_btn = gr.Button("Reset to First Unannotated")

        with gr.Row():
            stats_btn = gr.Button("Show Dataset Statistics")
            dataset_stats = gr.Markdown(value=preview_data(csv_file))

        annotate_btn.click(
            fn=lambda row_idx, label, vtype, trans, df: annotate_post(
                row_idx, label, vtype, trans, df, csv_file
            ),
            inputs=[row_idx_state, label_choice, type_choice, english_trans, df_state],
            outputs=[
                row_idx_state, df_state, post_number, russian_post,
                english_trans, status, label_choice, type_choice,
            ],
        )
        reset_btn.click(
            fn=lambda df: reset_annotation(df, csv_file),
            inputs=[df_state],
            outputs=[
                row_idx_state, df_state, post_number, russian_post,
                english_trans, status, label_choice, type_choice,
            ],
        )
        stats_btn.click(
            fn=lambda: preview_data(csv_file),
            outputs=dataset_stats,
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=share,
        debug=debug,
    )


if __name__ == "__main__":
    run_gradio_app()
