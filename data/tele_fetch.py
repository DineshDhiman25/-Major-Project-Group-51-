"""
data/tele_fetch.py
Fetches Telegram post texts for a list of (Channel ID, Post ID) pairs.
"""

from __future__ import annotations

import pandas as pd


def fetch_telegram_posts(
    input_csv: str,
    output_csv: str,
    username: str,
    api_id: int,
    api_hash: str,
) -> pd.DataFrame:
    """
    Fetch Telegram post texts for every (Channel ID, Post ID) pair in *input_csv*
    and save results to *output_csv*.

    Parameters
    ----------
    input_csv  : Path to a CSV with columns [Channel ID, Post ID, Date].
    output_csv : Destination CSV path for fetched posts.
    username   : Telegram account username (session name).
    api_id     : Telegram API ID.
    api_hash   : Telegram API hash.

    Returns
    -------
    DataFrame with columns [Channel ID, Post ID, Date, Post].
    """
    from telethon.sync import TelegramClient  # optional dependency

    df = pd.read_csv(input_csv)
    results: list[dict] = []

    with TelegramClient(username, api_id, api_hash) as client:
        for _, row in df.iterrows():
            ch = int(row["Channel ID"])
            post_id = int(row["Post ID"])
            channel = int("-100" + str(ch))

            try:
                print(f"Fetching post: Channel {channel} | Post ID {post_id}")
                msg = client.get_messages(channel, ids=post_id)

                if msg and msg.text:
                    text = msg.text
                else:
                    text = None
                    print("⚠ No text found.")

            except Exception as e:
                print(f"Error fetching {post_id} from {channel}: {e}")
                text = None

            results.append({
                "Channel ID": ch,
                "Post ID": post_id,
                "Date": row["Date"],
                "Post": text,
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(out_df)} rows to: {output_csv}")
    return out_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Telegram posts")
    parser.add_argument("--input",    required=True, help="Input CSV path")
    parser.add_argument("--output",   required=True, help="Output CSV path")
    parser.add_argument("--username", required=True)
    parser.add_argument("--api-id",   required=True, type=int)
    parser.add_argument("--api-hash", required=True)
    args = parser.parse_args()

    fetch_telegram_posts(
        input_csv=args.input,
        output_csv=args.output,
        username=args.username,
        api_id=args.api_id,
        api_hash=args.api_hash,
    )
