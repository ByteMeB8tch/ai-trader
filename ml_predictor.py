import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from logger_config import get_logger

logger = get_logger()

# --- Feature Engineering and Creation ---
def create_features_and_target(df):
    """
    Creates technical indicator features and target variable for ML training.
    Target: 1 if next minute's close is higher, else 0.
    """
    df_copy = df.copy()
    
    # Import add_indicators function from strategy module
    from strategy import add_indicators
    df_copy = add_indicators(df_copy)

    # Additional features
    df_copy['Price_Change'] = df_copy['Close'].pct_change()
    df_copy['High_Low_Ratio'] = (df_copy['High'] - df_copy['Low']) / df_copy['Close'].replace(0, 1)

    # Target: whether next close price is higher
    df_copy['Target'] = (df_copy['Close'].shift(-1) > df_copy['Close']).astype(int)

    # List of feature columns to use
    feature_columns = [
        'SMA9', 'SMA20', 'EMA12', 'EMA26',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'Volume_Ratio', 'ATR', 'Stoch_K', 'Stoch_D',
        'Price_Change', 'High_Low_Ratio', 'OBV',
        'ADX', 'CCI', 'KC_Upper', 'KC_Lower'
    ]

    # Drop rows with missing values (from indicator calculation)
    df_copy = df_copy.dropna()

    # Slice features and target
    features = df_copy[feature_columns]
    target = df_copy['Target']

    return features, target

# --- Train ML Model ---
def train_model(historical_df):
    """
    Train GradientBoostingClassifier model with historical data.
    Returns trained model or None on failure.
    """
    if historical_df.empty or len(historical_df) < 100:
        logger.warning("Insufficient historical data to train ML model.")
        return None

    features, target = create_features_and_target(historical_df)

    if features.empty or len(features) < 2:
        logger.warning("Not enough feature rows after preprocessing.")
        return None

    # Use non-shuffled split for time series like data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False)

    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"ML Model trained. Accuracy on test set: {accuracy:.2f}")

    return model

# --- Predict Next Movement ---
def predict_next_price_movement(model, current_df):
    """
    Predict price movement: 1 = price up next period, 0 = not up.
    Returns (prediction, probability_of_up).
    """
    if model is None:
        return -1, 0.0

    features, _ = create_features_and_target(current_df)
    if features.empty:
        return -1, 0.0

    latest_features = features.iloc[-1].values.reshape(1, -1)

    # Construct DataFrame with correct columns
    latest_features_df = pd.DataFrame(latest_features, columns=features.columns)

    prediction = model.predict(latest_features_df)[0]
    probability_of_up = model.predict_proba(latest_features_df)[0][1]

    return prediction, probability_of_up
