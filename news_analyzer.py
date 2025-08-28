import os
import time
import requests
import google.generativeai as genai
import json # Added for JSON parsing
from datetime import datetime, timedelta
from dotenv import load_dotenv
from logger_config import get_logger

load_dotenv()
logger = get_logger()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.error("GEMINI_API_KEY is not set. Gemini analysis will be skipped.")

# --- API Rate Limit Management ---
# Basic rate limiting for external APIs
NEWS_API_LAST_CALL_TIME = 0
NEWS_API_RATE_LIMIT_SECONDS = 10 # Example: 1 call every 10 seconds (adjust based on your News API plan)

GEMINI_API_LAST_CALL_TIME = 0
GEMINI_API_RATE_LIMIT_SECONDS = 5 # Example: 1 call every 5 seconds (adjust based on your Gemini plan)

def enforce_rate_limit(api_name):
    global NEWS_API_LAST_CALL_TIME, GEMINI_API_LAST_CALL_TIME
    if api_name == "news":
        elapsed = time.time() - NEWS_API_LAST_CALL_TIME
        if elapsed < NEWS_API_RATE_LIMIT_SECONDS:
            sleep_time = NEWS_API_RATE_LIMIT_SECONDS - elapsed
            logger.info(f"News API rate limit hit. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
        NEWS_API_LAST_CALL_TIME = time.time()
    elif api_name == "gemini":
        elapsed = time.time() - GEMINI_API_LAST_CALL_TIME
        if elapsed < GEMINI_API_RATE_LIMIT_SECONDS:
            sleep_time = GEMINI_API_RATE_LIMIT_SECONDS - elapsed
            logger.info(f"Gemini API rate limit hit. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
        GEMINI_API_LAST_CALL_TIME = time.time()

# --- News API Integration (using Finnhub as an example) ---
# You might need to adapt this if you choose a different News API
def fetch_news_for_symbol(symbol, limit=5):
    """Fetches recent news articles for a given stock symbol."""
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not set. Skipping news fetch.")
        return []

    enforce_rate_limit("news")
    try:
        # Using Finnhub API as an example for news
        # You'll need to adjust the URL and parsing if you use a different news API
        # Finnhub requires a 'from' and 'to' date for news
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        news_url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={yesterday}&to={today}&token={NEWS_API_KEY}"
        response = requests.get(news_url, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors
        news_data = response.json()
        
        articles = []
        for article in news_data[:limit]: # Limit to 'limit' articles
            articles.append({
                'headline': article.get('headline'),
                'summary': article.get('summary'),
                'url': article.get('url')
            })
        logger.info(f"Fetched {len(articles)} news articles for {symbol}.")
        return articles
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in fetch_news_for_symbol for {symbol}: {e}")
        return []

# --- Gemini API Integration ---
def analyze_with_gemini(symbol, articles, recent_df):
    """
    Uses Gemini to analyze news sentiment and verify trade signals.
    Returns:
        dict: {'confidence': float, 'sentiment': str, 'rationale': str, 
               'suggested_stop_loss_percent': float, 'suggested_take_profit_percent': float}
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set. Skipping Gemini analysis.")
        return {'confidence': 0.0, 'sentiment': 'neutral', 'rationale': 'Gemini API not configured.', 
                'suggested_stop_loss_percent': 0.0, 'suggested_take_profit_percent': 0.0}

    enforce_rate_limit("gemini")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20') # Use the specified model

        news_text = "\n".join([f"Headline: {a['headline']}\nSummary: {a['summary']}" for a in articles])
        
        # Prepare recent price data for Gemini
        recent_price_data = recent_df[['Open', 'High', 'Low', 'Close']].tail(10).to_string()

        prompt = f"""
        Analyze the following news articles and recent stock price data for {symbol}.
        
        News Articles:
        {news_text}

        Recent 10-minute Price Data (Open, High, Low, Close):
        {recent_price_data}

        Based on the news and recent price action, provide:
        1. A sentiment (Positive, Negative, Neutral) towards the stock.
        2. A confidence score (0.0 to 1.0) for a short-term profitable trade (next 1-5 minutes).
        3. A brief rationale for your sentiment and confidence score.
        4. Suggest a dynamic stop-loss percentage (e.g., 2.5) and take-profit percentage (e.g., 5.0) based on the confidence.
        
        Format your response as a JSON object:
        {{
            "sentiment": "...",
            "confidence": 0.0,
            "rationale": "...",
            "suggested_stop_loss_percent": 0.0,
            "suggested_take_profit_percent": 0.0
        }}
        """
        
        # Call Gemini API with exponential backoff
        retries = 0
        max_retries = 5
        while retries < max_retries:
            try:
                response = model.generate_content(prompt)
                gemini_output = response.text
                
                # Attempt to parse JSON. Gemini might sometimes return extra text.
                try:
                    parsed_response = json.loads(gemini_output)
                    # Validate keys and types from Gemini response
                    if not all(key in parsed_response for key in ['sentiment', 'confidence', 'rationale', 'suggested_stop_loss_percent', 'suggested_take_profit_percent']):
                        raise ValueError("Missing expected keys in Gemini response.")
                    if not isinstance(parsed_response['confidence'], (int, float)):
                        raise ValueError("Confidence score is not a number.")
                    if not isinstance(parsed_response['suggested_stop_loss_percent'], (int, float)):
                        raise ValueError("Suggested stop loss is not a number.")
                    if not isinstance(parsed_response['suggested_take_profit_percent'], (int, float)):
                        raise ValueError("Suggested take profit is not a number.")
                    
                    return parsed_response
                except json.JSONDecodeError:
                    # Try to extract JSON from a larger text if Gemini wraps it
                    start_idx = gemini_output.find('{')
                    end_idx = gemini_output.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = gemini_output[start_idx : end_idx + 1]
                        parsed_response = json.loads(json_str)
                        # Re-validate extracted JSON
                        if not all(key in parsed_response for key in ['sentiment', 'confidence', 'rationale', 'suggested_stop_loss_percent', 'suggested_take_profit_percent']):
                            raise ValueError("Missing expected keys in extracted Gemini JSON.")
                        if not isinstance(parsed_response['confidence'], (int, float)):
                            raise ValueError("Confidence score is not a number in extracted JSON.")
                        if not isinstance(parsed_response['suggested_stop_loss_percent'], (int, float)):
                            raise ValueError("Suggested stop loss is not a number in extracted JSON.")
                        if not isinstance(parsed_response['suggested_take_profit_percent'], (int, float)):
                            raise ValueError("Suggested take profit is not a number in extracted JSON.")
                        return parsed_response
                    else:
                        raise ValueError("Could not extract JSON from Gemini response.")

            except Exception as e:
                retries += 1
                sleep_time = 2 ** retries # Exponential backoff
                logger.warning(f"Gemini API call failed (retry {retries}/{max_retries}): {e}. Retrying in {sleep_time} seconds.")
                time.sleep(sleep_time)
        
        logger.error(f"Gemini API call failed after {max_retries} retries for {symbol}.")
        return {'confidence': 0.0, 'sentiment': 'neutral', 'rationale': 'Gemini API failed after retries.', 
                'suggested_stop_loss_percent': 0.0, 'suggested_take_profit_percent': 0.0}

    except Exception as e:
        logger.error(f"Error during Gemini analysis for {symbol}: {e}")
        return {'confidence': 0.0, 'sentiment': 'neutral', 'rationale': f'Analysis failed: {e}', 
                'suggested_stop_loss_percent': 0.0, 'suggested_take_profit_percent': 0.0}

