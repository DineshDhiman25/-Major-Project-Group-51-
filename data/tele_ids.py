import asyncio
from telethon import TelegramClient
from datetime import datetime
import random
import pandas as pd

api_id = ''
api_hash = ''
phone = ''
username = ''

async def extract_posts(client, channel_id, target_count_min=10000, target_count_max=30000):
    """
    Extract random post IDs from a Telegram channel before December 2021
    that contain media and text with more than 10 words.

    Args:
        client: Connected TelegramClient instance
        channel_id: Telegram channel ID (e.g., 1105313000)
        target_count_min: Minimum number of posts to extract (default: 1000)
        target_count_max: Maximum number of posts to extract (default: 1500)
    """
    # Prefix with -100 so Telethon resolves it as a channel/supergroup,
    # not a user. Bare ints are treated as user IDs by default.
    full_channel_id = int(f"-100{channel_id}")

    print(f"Fetching posts from channel {channel_id}...")

    # Get the channel entity
    try:
        channel = await client.get_entity(full_channel_id)
    except Exception as e:
        print(f"Error getting channel: {e}")
        return None

    matching_posts = []
    total_checked = 0

    # Target to collect 5000-10000 posts before random selection
    collection_target = 30000  # Will stop at 10000 or when messages run out

    # Iterate through messages
    async for message in client.iter_messages(channel):
        total_checked += 1

        # Check if message has a date (it should)
        if not message.date:
            continue

        # Check if message has media (photo or video)
        has_media = message.photo or message.video

        # Check if message has text with more than 10 words
        has_valid_text = False
        if message.text:
            word_count = len(message.text.split())
            has_valid_text = word_count > 13

        # If all criteria met, add to list
        if has_media and has_valid_text:
            matching_posts.append({
                'channel_id': channel_id,
                'post_id': message.id,
                'date': message.date,
                'text_preview': message.text,
            })

        # Progress update every 1000 messages
        if total_checked % 1000 == 0:
            print(f"Checked {total_checked} messages, found {len(matching_posts)} matching posts...")

        # Stop when we reach collection_target matching posts
        if len(matching_posts) >= collection_target:
            print(f"Reached collection target of {collection_target} matching posts, stopping...")
            break

    print(f"\nTotal messages checked: {total_checked}")
    print(f"Total matching posts found: {len(matching_posts)}")

    # Randomly select between target_count_min and target_count_max posts
    if len(matching_posts) < target_count_min:
        print(f"Warning: Only found {len(matching_posts)} matching posts, less than minimum {target_count_min}")
        selected_posts = matching_posts
    else:
        selected_posts = matching_posts

    # Create DataFrame
    df = pd.DataFrame(selected_posts)

    # Sort by date for easier reading
    if not df.empty:
        df = df.sort_values('date', ascending=False)
        df.reset_index(drop=True, inplace=True)

    return df

async def main():
    # Example: Extract posts from a channel
    channel_ids = [1005280116, 1002843824]

    # Create the client once and reuse it across all channels
    # to avoid repeated reconnects (which cause TimeoutErrors)
    async with TelegramClient(username, api_id, api_hash) as client:
        for channel_id in channel_ids:
            df = await extract_posts(client, channel_id, target_count_min=10000, target_count_max=30000)
            if df is not None and not df.empty:
                # Save to CSV
                output_file = f'telegram_posts_{channel_id}.csv'
                df.to_csv(output_file, index=False)
                print(f"\nResults saved to {output_file}")
            else:
                print("No matching posts found or error occurred.")

# Run the script
if __name__ == '__main__':
    asyncio.run(main())


#1233777422,1633131143,1413275904,1768609733,1080134301,1152463236,1351029634,1536211233,1247556894,1457892982,1487098807,1796510941,1231519967,1078868616,1103152034,1036240821,1254661214,1429590454,1280273449,1145781893,1124038902,1232032465,1469021333,1135021433,1164348791,1093357968,1109403194,1498939244,1074354585,1283359437,1511148555,1111348665,1144404150,1129015804,1366346975,1378437829,1476769133,1472280635,1319370046,1135966410,1297235221,1552544518,1144180066,1308224148,1437890164,1153178038,1576917998,1307886524,1478765631,1572748754,1343028414, 1407902266, 1463721328,
