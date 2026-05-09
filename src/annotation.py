"""
src/annotation.py
=================
Unified entry point for the HRV annotation pipeline.

Imports and re-exports public functions from:
  - data/tele_fetch.py    → fetch_telegram_posts
  - data/annotate_yes.py  → build_yes_jsonl, TYPE_MAP, INSTRUCTION_TEXT
  - data/annotate_no.py   → build_no_jsonl, NO_RESPONSE
  - data/Shuffle.py       → shuffle_jsonl
  - Annotation/app.py     → run_gradio_app

CLI usage:
    python src/annotation.py fetch   --input posts.csv --output fetched.csv \\
                                     --username me --api-id 123 --api-hash abc
    python src/annotation.py yes     --input hrv_yes.csv --output hrv_yes.jsonl
    python src/annotation.py no      --input hrv_no.csv  --output hrv_no.jsonl
    python src/annotation.py shuffle --files hrv_train.jsonl hrv_test.jsonl hrv_val.jsonl
    python src/annotation.py app     [--csv path/to/file.csv] [--port 7860] [--share]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "data"))
sys.path.insert(0, str(_ROOT / "Annotation"))

from tele_fetch import fetch_telegram_posts          # noqa: E402
from annotate_yes import (                           # noqa: E402
    build_yes_jsonl,
    INSTRUCTION_TEXT,
    TYPE_MAP,
)
from annotate_no import build_no_jsonl, NO_RESPONSE  # noqa: E402
from Shuffle import shuffle_jsonl                    # noqa: E402
from app import run_gradio_app                       # noqa: E402

__all__ = [
    "fetch_telegram_posts",
    "build_yes_jsonl",
    "build_no_jsonl",
    "shuffle_jsonl",
    "run_gradio_app",
    "INSTRUCTION_TEXT",
    "NO_RESPONSE",
    "TYPE_MAP",
]

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annotation",
        description="HRV Annotation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- fetch
    p_fetch = sub.add_parser("fetch", help="Fetch Telegram posts into a CSV")
    p_fetch.add_argument("--input",    required=True, help="Input CSV (Channel ID, Post ID, Date)")
    p_fetch.add_argument("--output",   required=True, help="Output CSV path")
    p_fetch.add_argument("--username", required=True, help="Telegram session username")
    p_fetch.add_argument("--api-id",   required=True, type=int, dest="api_id")
    p_fetch.add_argument("--api-hash", required=True, dest="api_hash")

    # -- yes
    p_yes = sub.add_parser("yes", help="Build YES fine-tune JSONL from annotated CSV")
    p_yes.add_argument("--input",  required=True, help="CSV with HRV-positive rows")
    p_yes.add_argument("--output", required=True, help="Output .jsonl path")

    # -- no
    p_no = sub.add_parser("no", help="Build NO fine-tune JSONL from annotated CSV")
    p_no.add_argument("--input",  required=True, help="CSV with HRV-negative rows")
    p_no.add_argument("--output", required=True, help="Output .jsonl path")

    # -- shuffle
    p_shuf = sub.add_parser("shuffle", help="Shuffle one or more JSONL files in-place")
    p_shuf.add_argument("--files", nargs="+", required=True, help="JSONL files to shuffle")
    p_shuf.add_argument("--seed",  type=int, default=42)

    # -- app
    p_app = sub.add_parser("app", help="Launch the Gradio annotation web app")
    p_app.add_argument("--csv",      default="Csv/hrv_train_venezuela.csv",
                       help="CSV to annotate")
    p_app.add_argument("--port",     type=int, default=7860)
    p_app.add_argument("--share",    action="store_true")
    p_app.add_argument("--no-debug", action="store_true", dest="no_debug")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "fetch":
        fetch_telegram_posts(
            input_csv=args.input,
            output_csv=args.output,
            username=args.username,
            api_id=args.api_id,
            api_hash=args.api_hash,
        )
    elif args.command == "yes":
        build_yes_jsonl(input_csv=args.input, output_path=args.output)
    elif args.command == "no":
        build_no_jsonl(input_csv=args.input, output_path=args.output)
    elif args.command == "shuffle":
        for path in args.files:
            shuffle_jsonl(input_path=path, output_path=path, seed=args.seed)
    elif args.command == "app":
        run_gradio_app(
            csv_file=args.csv,
            server_port=args.port,
            share=args.share,
            debug=not args.no_debug,
        )


if __name__ == "__main__":
    main()
