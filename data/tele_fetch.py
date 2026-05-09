from telethon.sync import TelegramClient
import pandas as pd

# Load CSV
df = pd.read_csv(INPUT_CSV)

results = []

with TelegramClient(username, api_id, api_hash) as client:
    for idx, row in df.iterrows():
        ch = int(row["Channel ID"])
        post_id = int(row["Post ID"])
        channel = int("-100" + str(ch))

        try:
            print(f"Fetching post: Channel {channel} | Post ID {post_id}")
            msg = client.get_messages(channel, ids=post_id)

            if msg and msg.text:
                results.append({
                    "Channel ID": ch,
                    "Post ID": post_id,
                    "Date": row["Date"],
                    "Post": msg.text
                })
            else:
                results.append({
                    "Channel ID": ch,
                    "Post ID": post_id,
                    "Date": row["Date"],
                    "Post": None
                })
                print("⚠ No text found.")

        except Exception as e:
            print(f"Error fetching {post_id} from {channel}: {e}")
            results.append({
                "Channel ID": ch,
                "Post ID": post_id,
                "Date": row["Date"],
                "Post": None
            })

out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)
