"""
main.py
=======
Interactive script launcher for the Major_Grp51 HRV pipeline.

Run:
    python main.py
or non-interactively:
    python main.py --script annotation --cmd app
    python main.py --script llama    --dataset iran
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


DATASETS = {
    "1": {
        "key": "combined",
        "label": "Combined  (src/hrv_train.jsonl + src/hrv_test.jsonl)",
    },
    "2": {
        "key": "iran",
        "label": "Iran      (data/jsonl/hrv_iran_train.jsonl + hrv_iran_test.jsonl)",
    },
    "3": {
        "key": "russia",
        "label": "Russia    (data/jsonl/hrv_rus_train.jsonl + hrv_rus_test.jsonl)",
    },
    "4": {
        "key": "venezuela",
        "label": "Venezuela (data/jsonl/hrv_vene_train.jsonl + hrv_vene_test.jsonl)",
    },
}

ANNOTATION_CMDS = {
    "1": ("app",     "Launch Gradio annotation web app"),
    "2": ("fetch",   "Fetch Telegram posts → CSV"),
    "3": ("yes",     "Build YES fine-tune JSONL from CSV"),
    "4": ("no",      "Build NO  fine-tune JSONL from CSV"),
    "5": ("shuffle", "Shuffle JSONL file(s)"),
}

LLM_SCRIPTS = {
    "1": {
        "key":    "llama",
        "script": "llm/finetuning_llama31_instruct.py",
        "label":  "Llama 3.1 8B Instruct",
    },
    "2": {
        "key":    "mistral",
        "script": "llm/finetuning_mistral_instruct.py",
        "label":  "Mistral 7B Instruct v0.3",
    },
    "3": {
        "key":    "openhermes",
        "script": "llm/finetuning_openhermes_mistral.py",
        "label":  "OpenHermes-2.5 Mistral",
    },
}



def _python() -> str:
    """Return the Python executable used to run this script."""
    return sys.executable


def _run(cmd: list[str]) -> None:
    print(f"\n▶  {' '.join(cmd)}\n{'='*60}")
    subprocess.run(cmd, cwd=ROOT)


def _pick(prompt: str, options: dict) -> str:
    """Display a numbered menu and return the chosen option key."""
    print(f"\n{prompt}")
    for k, v in options.items():
        label = v["label"] if isinstance(v, dict) else v[1]
        print(f"  {k}) {label}")
    while True:
        choice = input("Choice: ").strip()
        if choice in options:
            return choice
        print(f"  Invalid choice. Enter one of: {', '.join(options)}")


def run_annotation(cmd_key: str | None = None, extra_args: list[str] | None = None) -> None:
    """Launch src/annotation.py with a chosen sub-command."""
    if cmd_key is None:
        choice = _pick("Annotation sub-command:", ANNOTATION_CMDS)
        cmd_key = ANNOTATION_CMDS[choice][0]

    cmd = [_python(), str(ROOT / "src" / "annotation.py"), cmd_key]
    if extra_args:
        cmd += extra_args
    _run(cmd)


def run_llm(script_key: str | None = None, dataset_key: str | None = None) -> None:
    """Launch one of the llm/ fine-tuning scripts."""
    # Choose script
    if script_key is None:
        choice = _pick("LLM fine-tuning script:", LLM_SCRIPTS)
        script_info = LLM_SCRIPTS[choice]
    else:
        script_info = next(
            (v for v in LLM_SCRIPTS.values() if v["key"] == script_key), None
        )
        if script_info is None:
            print(f"Unknown script key '{script_key}'. Valid: {[v['key'] for v in LLM_SCRIPTS.values()]}")
            sys.exit(1)

    # Choose dataset
    if dataset_key is None:
        choice = _pick("Dataset:", DATASETS)
        dataset_key = DATASETS[choice]["key"]

    cmd = [_python(), str(ROOT / script_info["script"]), "--dataset", dataset_key]
    _run(cmd)


MAIN_MENU = {
    "1": ("annotation", "Annotation pipeline  (src/annotation.py)"),
    "2": ("llm",        "LLM fine-tuning      (llm/)"),
}


def interactive_menu() -> None:
    print("\n" + "=" * 60)
    print("  Major_Grp51 — HRV Pipeline Launcher")
    print("=" * 60)

    choice = _pick("Select pipeline:", MAIN_MENU)
    pipeline = MAIN_MENU[choice][0]

    if pipeline == "annotation":
        run_annotation()
    elif pipeline == "llm":
        run_llm()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="HRV Pipeline Launcher",
    )
    sub = p.add_subparsers(dest="pipeline")

    # annotation
    pa = sub.add_parser("annotation", help="Run annotation pipeline")
    pa.add_argument(
        "--cmd",
        choices=[v[0] for v in ANNOTATION_CMDS.values()],
        default="app",
        help="Annotation sub-command (default: app)",
    )

    # llm
    pl = sub.add_parser("llm", help="Run LLM fine-tuning")
    pl.add_argument(
        "--script",
        choices=[v["key"] for v in LLM_SCRIPTS.values()],
        default="llama",
        help="Model script to run (default: llama)",
    )
    pl.add_argument(
        "--dataset",
        choices=[v["key"] for v in DATASETS.values()],
        default="combined",
        help="Dataset to use (default: combined)",
    )

    return p


def main() -> None:
    parser = _build_parser()
    # If no sub-command given, fall back to interactive menu
    if len(sys.argv) == 1:
        interactive_menu()
        return

    args = parser.parse_args()

    if args.pipeline == "annotation":
        run_annotation(cmd_key=args.cmd)
    elif args.pipeline == "llm":
        run_llm(script_key=args.script, dataset_key=args.dataset)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
