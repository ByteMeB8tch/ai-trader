import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from logger_config import get_logger

logger = get_logger()

# --- Feature Engineering and Target Creation ---
def create_features_and_target(df):
    """
    Creates technical indicator features and a target variable for ML training.
    Target: 1 if next minute's close price is higher, 0 otherwise.
    """
    df_copy = df.copy()

    df_copy['SMA9'] = df_copy['Close'].rolling(window=9).mean()
    df_copy['SMA20'] = df_copy['Close'].rolling(window=20).mean()
    
    # Calculate RSI more robustly
    delta = df_copy['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()

    rs = avg_gain / avg_loss
    df_copy['RSI'] = 100 - (100 / (1 + rs)) 
    
    df_copy['RSI'] = df_copy['RSI'].replace([np.inf, -np.inf], np.nan)

    df_copy['Volume_Change'] = df_copy['Volume'].pct_change()
    df_copy['Price_Change'] = df_copy['Close'].pct_change()

    df_copy['Target'] = (df_copy['Close'].shift(-1) > df_copy['Close']).astype(int)

    df_copy = df_copy.dropna()
    
    features = df_copy[['SMA9', 'SMA20', 'RSI', 'Volume_Change', 'Price_Change']]
    target = df_copy['Target']
    
    return features, target

# --- Model Training ---
def train_model(historical_df):
    """
    Trains a Gradient Boosting Classifier model using historical data.
    Returns the trained model.
    """
    if historical_df.empty or len(historical_df) < 100:
        logger.warning("Insufficient historical data to train ML model.")
        return None

    features, target = create_features_and_target(historical_df.copy())

    if features.empty or len(features) < 2:
        logger.warning("Not enough features/target data after preprocessing for ML training.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=False)

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"ML Model trained. Accuracy on test set: {accuracy:.2f}")

    return model

# --- Prediction ---
def predict_next_price_movement(model, current_df):
    """
    Predicts the next price movement (1 for up, 0 for down/stable) using the trained model.
    Returns:
        tuple: (prediction, probability_of_up)
    """
    if model is None:
        return -1, 0.0

    features, _ = create_features_and_target(current_df.copy())
    
    if features.empty:
        return -1, 0.0

    latest_features = features.iloc[-1].values.reshape(1, -1)
    
    # Ensure the input to predict has the correct column names
    latest_features_df = pd.DataFrame(latest_features, columns=features.columns)

    prediction = model.predict(latest_features_df)[0]
    probability_of_up = model.predict_proba(latest_features_df)[0][1]

    return prediction, probability_of_up