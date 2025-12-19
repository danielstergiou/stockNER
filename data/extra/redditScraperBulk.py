import praw
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
import re
import html
import unicodedata

load_dotenv()

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', ' ', text)
    
    # Remove Reddit links
    text = re.sub(r'/r/\w+|/u/\w+', ' ', text)
    
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def scrape_subreddit(
    subreddit: str,
    mode: str = "new",              # "new" | "hot" | "top"
    limit: int = 1000,
    time_filter: str = "year",      # for "top": "day","week","month","year","all"
    min_text_length: int = 100,     # minimum characters after cleaning
    out_csv: str = None,
    out_txt: str = None,
):
    MASTER_TXT = "/Users/danielstergiou/Desktop/Projects/stockNER/data/redditDatasets/cleanedRedditDatasets/allRedditPosts.txt"
    
    if out_txt is None:
        out_txt = f"/Users/danielstergiou/Desktop/Projects/stockNER/data/redditDatasets/cleanedRedditDatasets/r-{subreddit.lower()}Posts.txt"
    
    print(f"\n{'='*60}")
    print(f"Scraping r/{subreddit}")
    print(f"{'='*60}")
    print(f"Mode: {mode} | Limit: {limit} | Min length: {min_text_length}")
    print(f"{'='*60}\n")
    
    # Connect to Reddit
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT'),
        username=os.getenv('REDDIT_USERNAME'),
        password=os.getenv('REDDIT_PASSWORD')
    )
    
    sub = reddit.subreddit(subreddit)

    # Get posts based on mode
    if mode == "new":
        posts = sub.new(limit=limit)
    elif mode == "hot":
        posts = sub.hot(limit=limit)
    elif mode == "top":
        posts = sub.top(time_filter=time_filter, limit=limit)
    else:
        raise ValueError("mode must be: new, hot, top")

    rows = []
    skipped_score = 0
    skipped_stickied = 0
    
    for p in posts:
        # Skip stickied posts (usually mod announcements)
        if p.stickied:
            skipped_stickied += 1
            continue
        
        # Combine and clean text
        raw_text = f"{(p.title or '').strip()} {(p.selftext or '').strip()}".strip()
        cleaned_text = clean_text(raw_text)
        
        created_utc = datetime.fromtimestamp(p.created_utc, tz=timezone.utc).isoformat()
        
        rows.append({
            "id": p.id,
            "subreddit": subreddit,
            "created_utc": created_utc,
            "title": p.title or "",
            "selftext": p.selftext or "",
            "text": cleaned_text,  # Cleaned text for NER
            "raw_text": raw_text,
            "score": p.score,
            "num_comments": p.num_comments,
            "url": p.url,
            "permalink": "https://www.reddit.com" + p.permalink,
        })

    df = pd.DataFrame(rows)
    
    # Remove duplicates by text content
    original_len = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    duplicates_removed = original_len - len(df)
    
    # Write per-subreddit file (overwrite is fine)
    with open(out_txt, 'w', encoding='utf-8') as f:
        for text in df['text']:
            f.write(text + '\n')

    # Append to master file
    with open(MASTER_TXT, 'a', encoding='utf-8') as f:
        for text in df['text']:
            f.write(text + '\n')

    
    # Print summary
    print(f"Results:")
    print(f"  ✓ Collected: {len(df)} posts")
    print(f"  ✗ Skipped (low score): {skipped_score}")
    print(f"  ✗ Skipped (stickied): {skipped_stickied}")
    print(f"  ✗ Duplicates removed: {duplicates_removed}")
    print(f"\nAverage text length: {df['text'].str.len().mean():.0f} characters")
    
    return df


# ============================================================================
# CONFIGURATION - Edit these settings for each subreddit
# ============================================================================

if __name__ == "__main__":
            
    scrape_subreddit(
        subreddit="wallstreetbets",
        mode="top",
        time_filter="year",
        limit=1000,
        min_text_length=15,
    )
    
    scrape_subreddit(
        subreddit="stocks",
        mode="top",
        time_filter="year",
        limit=700,
        min_text_length=15,
    )
    
    scrape_subreddit(
        subreddit="investing",
        mode="top",
        time_filter="year",
        limit=500,
        min_text_length=15,
    )
        
    scrape_subreddit(
        subreddit="economics",
        mode="top",
        time_filter="year",
        limit=500,
        min_text_length=15,
    )
    
    scrape_subreddit(
        subreddit="economy",
        mode="top",
        time_filter="year",
        limit=300,
        min_text_length=15,
    )
        
    scrape_subreddit(
        subreddit="ETFs",
        mode="top",
        time_filter="all",
        limit=400,
        min_text_length=15,
    )
    
    scrape_subreddit(
        subreddit="Bogleheads",
        mode="top",
        time_filter="year",
        limit=300,
        min_text_length=15,
    )
        
    scrape_subreddit(
        subreddit="Commodities",
        mode="top",
        time_filter="all",
        limit=400,
        min_text_length=15,
    )
        
    scrape_subreddit(
        subreddit="options",
        mode="top",
        time_filter="year",
        limit=400,
        min_text_length=15,
    )
    