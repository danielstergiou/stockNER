import praw, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT'),
            username=os.getenv('REDDIT_USERNAME'),
            password=os.getenv('REDDIT_PASSWORD')
)

subreddits = ["ETFs"]
#subreddits = ["Commodities"]
        
all_posts = []

for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    
    for post in subreddit.hot(limit=100):
        title = post.title.strip()
        body = post.selftext.strip()
        
        if not title and not body:
            continue  # skip empty posts

        combined_text = f"{title} {body}".strip()
        all_posts.append(combined_text)

df = pd.DataFrame(all_posts)
df.to_csv("r-etfs.txt", index=False)
