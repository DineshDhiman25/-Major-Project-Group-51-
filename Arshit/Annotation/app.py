import gradio as gr
import pandas as pd
import os

CSV_FILE = 'Csv/hrv_yes.csv'

def load_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=['Index', 'Post', 'Trans', 'Label', 'Type'])
    required_cols = ['Index', 'Post', 'Trans', 'Label', 'Type']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    next_row_idx = 0
    for idx, row in df.iterrows():
        if pd.isna(row.get('Label', None)) or row.get('Label', '') == '':
            next_row_idx = idx
            break
    else:
        next_row_idx = len(df)
    return next_row_idx, df

def save_data(df):
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    df.to_csv(CSV_FILE, index=False)
    return "Data saved successfully!"

def annotate_post(row_idx, label, violation_type, english_trans_text, df):
    if row_idx < len(df):
        df.at[row_idx, 'Label'] = label
        df.at[row_idx, 'Trans'] = english_trans_text
        if isinstance(violation_type, list):
            formatted = ', '.join([t[0] for t in violation_type])
            df.at[row_idx, 'Type'] = formatted
        else:
            df.at[row_idx, 'Type'] = violation_type[0] if violation_type else ""
        save_data(df)
        next_row = len(df)
        for idx, row in df.iterrows():
            if pd.isna(row.get('Label', None)) or row.get('Label', '') == '':
                next_row = idx
                break
        if next_row < len(df):
            current_row = df.iloc[next_row]
            post_display = f"Post {current_row.get('Index', next_row + 1)}"
            russian_post = current_row.get('Post', '')
            english_trans_text = current_row.get('Trans', '')
            status = "Next post loaded"
        else:
            post_display = "All posts annotated!"
            russian_post = ""
            english_trans_text = ""
            status = "All posts have been annotated. Great job!"
        return (
            next_row, df, post_display, russian_post, english_trans_text, status,
            None, None
        )
    else:
        return (
            row_idx, df, "No more posts to annotate", "", "", "Completed",
            None, None
        )


def get_type_choices():
    return [
        "A: Killing or injury of civilians",
        "B: Destruction of civilian objects (homes, hospitals, schools, markets)",
        "C: Rape, torture, or execution",
        "D: Mistreatment or torture of prisoners/detainees"
    ]

def reset_annotation(df):
    row_idx, new_df = load_data()
    current_row = new_df.iloc[row_idx] if row_idx < len(new_df) else None
    if current_row is not None:
        post_display = f"Post {current_row.get('Index', row_idx + 1)}"
        russian_post = current_row.get('Post', '')
        english_trans_text = current_row.get('Trans', '')
        status = "Reset to first unannotated post"
    else:
        post_display = "No posts to annotate"
        russian_post = ""
        english_trans_text = ""
        status = "No posts available"
    return (
        row_idx, new_df, post_display, russian_post, english_trans_text, status,
        None, None
    )


def preview_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        total_posts = len(df)
        annotated = len(df[df['Label'].notna() & (df['Label'] != '')])
        unannotated = total_posts - annotated
        types_count = df['Type'].value_counts().to_dict() if 'Type' in df.columns else {}
        stats = f"""
        **Dataset Statistics**
        - Total Posts: {total_posts}
        - Annotated: {annotated}
        - Remaining: {unannotated}

        **Type Distribution:**
        """
        for type_key, count in types_count.items():
            if pd.notna(type_key) and type_key:
                stats += f"\n- {type_key}: {count}"
        return stats
    return "No data file found."

initial_row_idx, initial_df = load_data()
if initial_row_idx < len(initial_df):
    initial_row = initial_df.iloc[initial_row_idx]
    initial_post_display = f"Post {initial_row.get('Index', initial_row_idx + 1)}"
    initial_russian = initial_row.get('Post', '')
    initial_english = initial_row.get('Trans', '')
    initial_status = "Ready to annotate"
else:
    initial_post_display = "No posts to annotate"
    initial_russian = ""
    initial_english = ""
    initial_status = "No data available"

with gr.Blocks(title="Human Rights Violation Annotation Tool") as demo:
    gr.Markdown("# Human Rights Violation Annotation Tool")
    gr.Markdown("""
    Annotate Russian social media posts for human rights violations.
    Destruction of Military facilities is not HRV"
    """)

    row_idx_state = gr.State(value=initial_row_idx)
    df_state = gr.State(value=initial_df)

    with gr.Row():
        with gr.Column(scale=2):
            post_number = gr.Textbox(
                value=initial_post_display,
                label="Current Post",
                interactive=False
            )
            russian_post = gr.Textbox(
                value=initial_russian,
                label="Russian Post",
                lines=4,
                interactive=False,
                show_copy_button=True
            )

            english_trans = gr.Textbox(
                value=initial_english,
                label="English Translation",
                lines=4,
                interactive=True
            )

            status = gr.Textbox(
                value=initial_status,
                label="Status",
                interactive=False
            )

        with gr.Column(scale=1):
            gr.Markdown("### Annotation")

            label_choice = gr.Radio(
                choices=["Yes", "No"],
                label="Is this a Human Rights Violation?",
                value=None
            )
            type_choice = gr.CheckboxGroup(
                choices=get_type_choices(),
                label="Violation Type",
                value=None
            )

            annotate_btn = gr.Button("Save Annotation & Next Post", variant="primary")
            reset_btn = gr.Button("Reset to First Unannotated")
    with gr.Row():
        stats_btn = gr.Button("Show Dataset Statistics")
        dataset_stats = gr.Markdown(value=preview_data())
    annotate_btn.click(
        annotate_post,
        inputs=[row_idx_state, label_choice, type_choice, english_trans, df_state],
        outputs=[row_idx_state, df_state, post_number, russian_post, english_trans, status, label_choice, type_choice]
    )


    reset_btn.click(
        reset_annotation,
        inputs=[df_state],
        outputs=[row_idx_state, df_state, post_number, russian_post, english_trans, status, label_choice, type_choice]
    )
    stats_btn.click(
        preview_data,
        outputs=dataset_stats
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
