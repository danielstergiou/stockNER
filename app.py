import asyncio
import logging
import os
import csv
import re
from dotenv import load_dotenv
from datetime import datetime
import pytz
import pandas as pd
import torch
import asyncpraw
import spacy
from spacy.pipeline import EntityRuler
from transformers import BertTokenizer, BertForSequenceClassification
import nltk

# --- INITIAL SETUP ---

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LOAD NLP MODELS ---

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load FinBERT model and tokenizer for sentiment analysis
# Using a try-except block for robustness in case of model loading issues
try:
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    logging.info("FinBERT model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading FinBERT model: {e}")
    # Exit if the core model can't be loaded
    exit()

# Download NLTK data for sentence tokenization if not present
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# --- DATA LOADING AND PREPARATION ---

def load_ticker_data(stocks_csv_path, etfs_csv_path):
    """
    Loads ticker, name, and sector data from CSV files into structured maps and sets.
    
    Returns:
        tuple: A tuple containing:
            - ticker_info_map (dict): Maps a ticker to its name and sector.
            - valid_tickers (set): A set of all valid uppercase ticker symbols for fast lookups.
            - company_to_ticker_map (dict): Maps a full company name to its ticker.
    """
    df_stocks = pd.read_csv(stocks_csv_path)
    df_etfs = pd.read_csv(etfs_csv_path)
    df = pd.concat([df_stocks, df_etfs], ignore_index=True)
    df.fillna('', inplace=True)

    ticker_info_map = {}
    valid_tickers = set()
    company_to_ticker_map = {}

    for _, row in df.iterrows():
        ticker = row['Ticker'].strip().upper()
        name = row['Name'].strip().upper()
        sector = row['Sector'].strip()

        if not ticker:
            continue

        # Primary mapping from ticker to its info
        ticker_info_map[ticker] = {'name': name, 'sector': sector}
        valid_tickers.add(ticker)
        
        # Mapping from company name to its ticker for easy lookup
        if name:
            company_to_ticker_map[name] = ticker

    logging.info(f"Loaded {len(valid_tickers)} tickers and {len(company_to_ticker_map)} company names.")
    return ticker_info_map, valid_tickers, company_to_ticker_map

# Load all ticker-related data
ticker_info_map, valid_tickers, company_to_ticker_map = load_ticker_data(
    'data/updated_stockslist.csv', 
    'data/updated_etfs.csv'
)

def create_entity_ruler(nlp, company_map):
    """
    Creates and adds a spaCy EntityRuler to the NLP pipeline.
    The ruler uses company names to identify 'ORG' entities.
    """
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
        
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [{"label": "ORG", "pattern": name} for name in company_map.keys()]
    ruler.add_patterns(patterns)
    logging.info("spaCy EntityRuler created with company name patterns.")
    return ruler

# Add the custom entity ruler to the spaCy pipeline
create_entity_ruler(nlp, company_to_ticker_map)

# --- SENTIMENT ANALYSIS ---

def get_sentence_sentiment(text):
    """
    Analyzes the sentiment of a given text using FinBERT.
    
    Returns:
        dict: A dictionary with sentiment labels and their scores.
    """
    if not text:
        return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # FinBERT labels: 0 -> Positive, 1 -> Neutral, 2 -> Negative
    sentiment_scores = {
        "positive": predictions[0][0].item(),
        "neutral": predictions[0][1].item(),
        "negative": predictions[0][2].item()
    }
    return sentiment_scores

# --- CORE TICKER EXTRACTION LOGIC ---

# Set of common words that might be confused with tickers.
# This list helps in deciding when to apply stricter contextual checks.
AMBIGUOUS_TICKERS = {"A", "GO", "FOR", "SEE", "IT", "ON", "BE", "ALL", "NOW", "SO", "DD", "WS", "U"} & valid_tickers

# Financial keywords used for contextual validation
FINANCIAL_KEYWORDS = {
    "stock", "shares", "price", "market", "trade", "buy", "sell", "invest", "investment", 
    "ticker", "ipo", "dividend", "portfolio", "equity", "etf", "earnings", "profit", "loss",
    "long", "short", "call", "put", "bullish", "bearish", "hodl", "yolo"
}

def is_in_financial_context(token):
    """
    Checks if a token is grammatically linked to a financial keyword using dependency parsing.
    This is used to validate ambiguous tickers.
    """
    # Check the token's head (the word it depends on)
    if token.head.text.lower() in FINANCIAL_KEYWORDS:
        return True
    
    # Check the token's children (words that depend on it)
    for child in token.children:
        if child.text.lower() in FINANCIAL_KEYWORDS:
            return True
            
    return False

def extract_tickers_from_text(doc):
    """
    Extracts tickers from a spaCy Doc object using a multi-pass strategy.
    
    Returns:
        set: A set of unique, validated ticker symbols found in the text.
    """
    found_tickers = set()

    # Pass 1: High-precision cashtag extraction ($TICKER)
    cashtag_pattern = r'\$[A-Z]{1,5}\b'
    for match in re.finditer(cashtag_pattern, doc.text):
        ticker = match.group(0)[1:]  # Remove the '$'
        if ticker in valid_tickers:
            found_tickers.add(ticker)

    # Pass 2: NER for company names and unambiguous tickers
    for ent in doc.ents:
        # If a company name is found, map it back to its ticker
        if ent.label_ == "ORG" and ent.text.upper() in company_to_ticker_map:
            found_tickers.add(company_to_ticker_map[ent.text.upper()])
        # If an uppercase word is a valid ticker and not ambiguous
        elif (ent.text.upper() in valid_tickers and 
              ent.text.upper() not in AMBIGUOUS_TICKERS and 
              ent.text.isupper()):
            found_tickers.add(ent.text.upper())
            
    # Pass 3: Contextual validation for ambiguous tickers
    for token in doc:
        token_upper = token.text.upper()
        # Check if the token is an ambiguous ticker and hasn't been found yet
        if token_upper in AMBIGUOUS_TICKERS and token_upper not in found_tickers:
            # Use dependency parsing to confirm financial context
            if is_in_financial_context(token):
                found_tickers.add(token_upper)

    return found_tickers

# --- REDDIT STREAM PROCESSING ---

async def process_submission(submission):
    """
    Processes a single Reddit submission to extract tickers and their sentence-level sentiment.
    """
    title = submission.title
    text = submission.selftext
    full_text = title + " " + text
    
    # Process the text with spaCy once
    doc = nlp(full_text)
    
    # Extract all unique tickers from the text
    extracted_tickers = extract_tickers_from_text(doc)
    
    if not extracted_tickers:
        return # No tickers found, nothing more to do

    logging.info(f"--- Post ID: {submission.id} ---")
    logging.info(f"Found tickers: {', '.join(extracted_tickers)}")

    # Data structure to hold the results
    mention_results = {
        'post_id': submission.id,
        'post_url': f"https://reddit.com{submission.permalink}",
        'post_timestamp': datetime.fromtimestamp(submission.created_utc, tz=pytz.UTC),
        'mentions': []
    }

    # Analyze sentiment at the sentence level for each ticker
    for sentence in doc.sents:
        sentence_text = sentence.text.strip()
        if not sentence_text:
            continue
            
        # Find which of the extracted tickers are in this sentence
        tickers_in_sentence = {ticker for ticker in extracted_tickers if ticker in sentence.text.upper()}

        if tickers_in_sentence:
            # Get sentiment for this specific sentence
            sentiment = get_sentence_sentiment(sentence_text)
            
            for ticker in tickers_in_sentence:
                mention_data = {
                    'ticker': ticker,
                    'ticker_info': ticker_info_map.get(ticker, {}),
                    'sentence': sentence_text,
                    'sentiment': sentiment
                }
                mention_results['mentions'].append(mention_data)

    # For now, we will just log the results.
    # In a real application, you would save this to a database or file.
    if mention_results['mentions']:
        logging.info(f"Processed Mentions for Post {submission.id}:")
        for mention in mention_results['mentions']:
            logging.info(f"  - Ticker: {mention['ticker']}, Sentence: '{mention['sentence']}', Sentiment: {mention['sentiment']}")
    logging.info(f"--- End Post ID: {submission.id} ---\n")


async def main():
    """
    Main function to set up Reddit stream and process submissions.
    """
    try:
        reddit = asyncpraw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT'),
            username=os.getenv('REDDIT_USERNAME'),
            password=os.getenv('REDDIT_PASSWORD')
        )
        subreddit = await reddit.subreddit("wallstreetbets")
        logging.info("Successfully connected to Reddit and listening to r/wallstreetbets stream.")
    except Exception as e:
        logging.error(f"Failed to connect to Reddit: {e}")
        return

    # Process submissions from the stream
    post_limit = 20 # Limit for this example run
    count = 0
    async for submission in subreddit.stream.submissions(skip_existing=True):
        if count >= post_limit:
            logging.info(f"Reached post limit of {post_limit}. Shutting down.")
            break
        
        try:
            await process_submission(submission)
            count += 1
        except Exception as e:
            # Log the error but continue processing other submissions
            logging.error(f"An error occurred while processing submission {submission.id}: {e}", exc_info=True)

    await reddit.close()

if __name__ == "__main__":
    asyncio.run(main())