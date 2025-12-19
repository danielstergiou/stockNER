import pandas as pd
import re

# df = pd.read_csv('./data/stockslist.csv')
# pd.set_option('display.max_rows', None)

# df = df.map(lambda x: x.replace('"', '') if isinstance(x, str) else x)

# cleaned = df.sort_values(by=["Ticker"],ignore_index=True)
# cleaned.to_csv('./data/stockslist.csv',index=False)


def clean_and_prepare_dataset(file_path, text_column, title_column):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=[text_column, title_column])
    df[text_column] = df[text_column].astype(str)
    df[title_column] = df[title_column].astype(str)
    
    # Define the pattern to exclude daily discussion threads
    exclude_pattern = re.compile(r'daily discussion thread for \w+ \d{2}, \d{4}', re.IGNORECASE)
    df = df[~df[title_column].str.contains(exclude_pattern)]
    
    df['combined_text'] = df[title_column] + " " + df[text_column]
    
    def clean_text(text):
        emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"
                                    u"\U0001F300-\U0001F5FF"
                                    u"\U0001F680-\U0001F6FF"
                                    u"\U0001F700-\U0001F77F"
                                    u"\U0001F780-\U0001F7FF"
                                    u"\U0001F800-\U0001F8FF"
                                    u"\U0001F900-\U0001F9FF"
                                    u"\U0001FA00-\U0001FA6F"
                                    u"\U0001FA70-\U0001FAFF"
                                    u"\U00002702-\U000027B0"
                                    "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[\n\'",‘’“”]', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s$]', '', text)  # Preserve $ symbol
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    df['combined_text'] = df['combined_text'].apply(clean_text)
    df = df.drop_duplicates(subset=['combined_text'])
    df = df.drop(['score', 'id', 'url', 'comms_num', 'created', 'timestamp'], axis=1, errors='ignore')
    # df = df.drop(["id","author","created","retrieved","edited","pinned","archived","locked","removed","deleted","is_self","is_video","is_original_content","link_flair_text","upvote_ratio","score","gilded","total_awards_received","num_comments","num_crossposts","thumbnail","shortlink"], axis=1, errors='ignore')
    sampled_df = df.sample(n=1000, random_state=None)
    
    return sampled_df

if __name__ == "__main__":
    file_path = "/Users/danielstergiou/Desktop/Projects/stockNER/data/redditDatasets/r-wsb.csv"
    text_column = "body"
    title_column = "title"

    prepared_df = clean_and_prepare_dataset(file_path, text_column, title_column)

    out_path = "/Users/danielstergiou/Desktop/Projects/stockNER/data/redditDatasets/cleanedRedditDatasets/r-wsb1000.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in prepared_df["combined_text"]:
            f.write(entry + "\n")

    print(f"Done. Wrote {len(prepared_df)} lines to {out_path}")


#prepared_df.to_json('prepared_dataset_for_labeling.json', orient='records', lines=True)
# data source --> https://www.kaggle.com/datasets/gpreda/wallstreetbets-2022 