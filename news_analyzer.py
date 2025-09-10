import os
import time
import requests
import google.generativeai as genai
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from logger_config import get_logger

load_dotenv()
logger = get_logger()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.error("GEMINI_API_KEY is not set. Gemini analysis will be skipped.")

NEWS_API_LAST_CALL_TIME = 0
NEWS_API_RATE_LIMIT_SECONDS = 10

GEMINI_API_LAST_CALL_TIME = 0
GEMINI_API_RATE_LIMIT_SECONDS = 5

PERPLEXITY_API_LAST_CALL_TIME = 0
PERPLEXITY_API_RATE_LIMIT_SECONDS = 5

def enforce_rate_limit(api_name):
    global NEWS_API_LAST_CALL_TIME, GEMINI_API_LAST_CALL_TIME, PERPLEXITY_API_LAST_CALL_TIME
    current_time = time.time()
    if api_name == "news":
        elapsed = current_time - NEWS_API_LAST_CALL_TIME
        if elapsed < NEWS_API_RATE_LIMIT_SECONDS:
            sleep_time = NEWS_API_RATE_LIMIT_SECONDS - elapsed
            logger.info(f"News API rate limit hit. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
        NEWS_API_LAST_CALL_TIME = time.time()
    elif api_name == "gemini":
        elapsed = current_time - GEMINI_API_LAST_CALL_TIME
        if elapsed < GEMINI_API_RATE_LIMIT_SECONDS:
            sleep_time = GEMINI_API_RATE_LIMIT_SECONDS - elapsed
            logger.info(f"Gemini API rate limit hit. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
        GEMINI_API_LAST_CALL_TIME = time.time()
    elif api_name == "perplexity":
        elapsed = current_time - PERPLEXITY_API_LAST_CALL_TIME
        if elapsed < PERPLEXITY_API_RATE_LIMIT_SECONDS:
            sleep_time = PERPLEXITY_API_RATE_LIMIT_SECONDS - elapsed
            logger.info(f"Perplexity API rate limit hit. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
        PERPLEXITY_API_LAST_CALL_TIME = time.time()

def fetch_news_for_symbol(symbol, limit=5):
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not set. Skipping news fetch.")
        return []
    enforce_rate_limit("news")
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        news_url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={yesterday}&to={today}&token={NEWS_API_KEY}"
        response = requests.get(news_url, timeout=10)
        response.raise_for_status()
        news_data = response.json()
        articles = []
        for article in news_data[:limit]:
            articles.append({
                'headline': article.get('headline'),
                'summary': article.get('summary'),
                'url': article.get('url')
            })
        logger.info(f"Fetched {len(articles)} news articles for {symbol}.")
        return articles
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return []

def analyze_with_gemini(symbol, articles, recent_df, risk_multiplier=1.0):
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set. Skipping Gemini analysis.")
        return {'confidence': 0.0, 'sentiment': 'neutral', 'rationale': 'Gemini API not configured.', 'suggested_stop_loss_percent': 0.0, 'suggested_take_profit_percent': 0.0}
    enforce_rate_limit("gemini")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        news_text = "\n".join([f"Headline: {a['headline']}\nSummary: {a['summary']}" for a in articles])
        recent_price_data = recent_df[['Open', 'High', 'Low', 'Close']].tail(10).to_string()
        caution_msg = ""
        if risk_multiplier > 1.0:
            caution_msg = (f"IMPORTANT: Trading risk multiplier is currently {risk_multiplier:.2f}, meaning much more capital will be committed. "
                           "Be significantly more cautious, only recommend 'Positive' if trade is exceptionally strong, "
                           "and increase confidence threshold. Avoid trades likely to cause losses.")
        else:
            caution_msg = "Standard trading caution applies."

        prompt = f"""
Analyze the following news articles and recent stock price data for {symbol}.
{caution_msg}
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
        retries = 0
        max_retries = 5
        while retries < max_retries:
            try:
                response = model.generate_content(prompt)
                gemini_output = response.text
                try:
                    parsed_response = json.loads(gemini_output)
                except json.JSONDecodeError:
                    start_idx = gemini_output.find('{')
                    end_idx = gemini_output.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = gemini_output[start_idx:end_idx]
                        parsed_response = json.loads(json_str)
                    else:
                        raise ValueError("Could not extract JSON from Gemini response")
                required_keys = ['sentiment', 'confidence', 'rationale', 'suggested_stop_loss_percent', 'suggested_take_profit_percent']
                if not all(k in parsed_response for k in required_keys):
                    raise ValueError("Missing keys in Gemini response")
                if not isinstance(parsed_response['confidence'], (int, float)):
                    raise ValueError("Confidence not a number")
                if not isinstance(parsed_response['suggested_stop_loss_percent'], (int, float)):
                    raise ValueError("Stop loss not a number")
                if not isinstance(parsed_response['suggested_take_profit_percent'], (int, float)):
                    raise ValueError("Take profit not a number")
                return parsed_response
            except Exception as err:
                retries += 1
                sleep_time = 2 ** retries
                logger.warning(f"Gemini API call failed (try {retries}/{max_retries}): {err}. Retrying in {sleep_time}s.")
                time.sleep(sleep_time)
        logger.error(f"Gemini API failed after {max_retries} retries for {symbol}.")
        return {'confidence': 0.0, 'sentiment': 'neutral', 'rationale': 'Gemini API failed after retries.', 'suggested_stop_loss_percent': 0.0, 'suggested_take_profit_percent': 0.0}
    except Exception as e:
        logger.error(f"Error during Gemini analysis for {symbol}: {e}")
        return {'confidence': 0.0, 'sentiment': 'neutral', 'rationale': f'Analysis failed: {e}', 'suggested_stop_loss_percent': 0.0, 'suggested_take_profit_percent': 0.0}

def analyze_with_perplexity(text, risk_multiplier=1.0):
    if not PERPLEXITY_API_KEY:
        logger.warning("PERPLEXITY_API_KEY not set. Skipping Perplexity analysis.")
        return {'sentiment': 'neutral', 'confidence': 0.0, 'rationale': 'Perplexity API not configured'}
    enforce_rate_limit("perplexity")
    caution_msg = ""
    if risk_multiplier > 1.0:
        caution_msg = (f"IMPORTANT: Trading risk multiplier is currently {risk_multiplier:.2f}, "
                       "so be much stricter before calling any trade 'Positive'. Only recommend buying if the news indicates high probability of profit.")
    else:
        caution_msg = "Standard caution applies."
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": f"{caution_msg}\nAnalyze the sentiment of the following stock news for trading purposes as Positive, Negative, or Neutral:\n{text}",
            "max_tokens": 150
        }
        response = requests.post("https://api.perplexity.ai/v1/answers", headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        answer_text = result.get("answer", "").lower()
        sentiment = "neutral"
        if "positive" in answer_text:
            sentiment = "Positive"
        elif "negative" in answer_text:
            sentiment = "Negative"
        confidence = 0.7 if sentiment != "neutral" else 0.5
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "rationale": "Derived from Perplexity API"
        }
    except Exception as e:
        logger.error(f"Perplexity API call failed: {e}")
        return {'sentiment': 'neutral', 'confidence': 0.0, 'rationale': f'Perplexity call failed: {e}'}
def analyze_news_for_symbol(symbol, recent_df, risk_multiplier=1.0):
    articles = fetch_news_for_symbol(symbol, limit=5)
    if not articles:
        logger.info(f"No news articles found for {symbol}. Skipping analysis.")
        return None
    gemini_analysis = analyze_with_gemini(symbol, articles, recent_df, risk_multiplier)
    news_text = "\n".join([f"Headline: {a['headline']}\nSummary: {a['summary']}" for a in articles])
    perplexity_analysis = analyze_with_perplexity(news_text, risk_multiplier)
    combined_confidence = (gemini_analysis.get('confidence', 0.0) + perplexity_analysis.get('confidence', 0.0)) / 2
    combined_sentiment = gemini_analysis.get('sentiment', 'neutral')
    if gemini_analysis.get('sentiment') != perplexity_analysis.get('sentiment'):
        combined_sentiment = 'neutral'
    rationale = f"Gemini: {gemini_analysis.get('rationale', '')} | Perplexity: {perplexity_analysis.get('rationale', '')}"
    stop_loss = gemini_analysis.get('suggested_stop_loss_percent', 0.0)
    take_profit = gemini_analysis.get('suggested_take_profit_percent', 0.0)
    analysis_result = {
        'symbol': symbol,
        'sentiment': combined_sentiment,
        'confidence': combined_confidence,
        'rationale': rationale,
        'suggested_stop_loss_percent': stop_loss,
        'suggested_take_profit_percent': take_profit,
        'articles': articles
    }
    logger.info(f"Completed news analysis for {symbol}: Sentiment={combined_sentiment}, Confidence={combined_confidence:.2f}")
    return analysis_result
