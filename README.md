# AI-Trader: An Advanced Algorithmic Trading Bot

This project is an advanced algorithmic trading bot built with Python and Docker. It connects to the Alpaca paper trading API to execute an intelligent, multi-layered trading strategy based on a combination of machine learning predictions, AI-driven news analysis, and technical indicators.

The bot is designed for intraday trading, focusing on capital preservation and seeking high-probability trade opportunities.

## 🚀 Key Features

* **Multi-Stage Screening:** Efficiently scans a large universe of stocks (up to 1,000) in batches to identify the most promising technical candidates.
* **Machine Learning Prediction:** Uses a **Logistic Regression model** to predict the probability of a stock's price moving up or down in the near future. This moves the bot from a reactive to a predictive strategy.
* **AI-Powered Analysis with Gemini:** For the top trading candidates, the bot performs a deep-dive analysis by:
    * Fetching real-time news articles from Finnhub.
    * Sending the news and recent price data to the **Gemini API** for a comprehensive sentiment and confidence analysis.
    * Using Gemini's feedback to confirm the trade signal, adjust trade size, and set dynamic risk parameters.
* **Dynamic Risk Management:**
    * **Wallet-Based Sizing:** Calculates the ideal trade size based on a percentage of your total account equity, ensuring you don't over-commit to a single trade.
    * **Fixed Stop-Loss:** A mandatory 8% stop-loss is applied to every trade to prevent significant losses.
    * **AI-Adjusted Take-Profit:** Gemini provides dynamic take-profit suggestions that are used to lock in gains efficiently.
* **Intelligent Position Management:** The bot doesn't just buy and forget. It continuously monitors open positions, and if a held stock's outlook turns negative (based on Gemini's analysis), it will proactively sell the position to mitigate potential losses.
* **Intraday Focus:** The bot is designed to trade on minute-based data and automatically closes all open positions before 11 PM IST to avoid overnight market risk.

## ⚙️ Setup and Configuration

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/ai-trader.git](https://github.com/your-username/ai-trader.git)
    cd ai-trader
    ```

2.  **Create `.env` File:**
    Copy the `.env.example` file to `.env` and fill in your API keys for Alpaca, Finnhub, and Gemini.

    ```
    # Example .env content
    APCA_API_KEY_ID=...
    APCA_API_SECRET_KEY=...
    NEWS_API_KEY=...
    GEMINI_API_KEY=...
    ...
    ```

3.  **Build the Docker Image:**
    ```bash
    docker build -t ai-trader .
    ```

4.  **Run the Bot:**
    ```bash
    docker run --env-file .env ai-trader
    ```

## ⚠️ Important Notes

* This bot is intended for **paper trading** only. Do not use this code for live trading without thorough backtesting and a full understanding of the risks involved.
* The performance of the bot is heavily dependent on the quality of the data from the news and Gemini APIs, as well as the volatility of the market.
* The `open_positions_metadata` is an in-memory dictionary. If the bot's Docker container is stopped or restarted, this data will be lost. For production use, a persistent database solution would be required.