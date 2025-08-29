import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from logger_config import get_logger

logger = get_logger()

# --- Feature Engineering and Target Creation ---
def create_features_and_target(df):
    """
    Creates technical indicator features and a target variable for ML training.
    Target: 1 if next minute's close price is higher, 0 otherwise.
    """
    # Work on a copy to avoid SettingWithCopyWarning/FutureWarning
    df_copy = df.copy()

    df_copy['SMA9'] = df_copy['Close'].rolling(window=9).mean()
    df_copy['SMA20'] = df_copy['Close'].rolling(window=20).mean()
    
    # Calculate RSI more robustly to avoid inplace issues
    delta = df_copy['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()

    rs = avg_gain / avg_loss
    df_copy['RSI'] = 100 - (100 / (1 + rs)) 
    
    # Clean up any remaining inf/-inf from RSI calc before final dropna
    df_copy['RSI'].replace([np.inf, -np.inf], np.nan, inplace=True) # Ensure inplace=True is used correctly here

    df_copy['Volume_Change'] = df_copy['Volume'].pct_change()
    df_copy['Price_Change'] = df_copy['Close'].pct_change()

    # Target: 1 if next close is higher, 0 otherwise
    df_copy['Target'] = (df_copy['Close'].shift(-1) > df_copy['Close']).astype(int)

    # Drop NaN values created by rolling windows and shifting
    df_copy = df_copy.dropna()
    
    features = df_copy[['SMA9', 'SMA20', 'RSI', 'Volume_Change', 'Price_Change']]
    target = df_copy['Target']
    
    return features, target

# --- Model Training ---
def train_model(historical_df):
    """
    Trains a Logistic Regression model using historical data.
    Returns the trained model.
    """
    if historical_df.empty or len(historical_df) < 100: # Need sufficient data for training
        logger.warning("Insufficient historical data to train ML model.")
        return None

    features, target = create_features_and_target(historical_df.copy()) # Pass a copy to feature creation

    if features.empty or len(features) < 2:
        logger.warning("Not enough features/target data after preprocessing for ML training.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=False)

    model = LogisticRegression(solver='liblinear', random_state=42) # Using liblinear for smaller datasets
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
        return -1, 0.0 # No prediction if no model

    # Create features for the current data point
    # Ensure current_df is also processed to create features
    features, _ = create_features_and_target(current_df.copy())
    
    if features.empty:
        return -1, 0.0

    # Get the latest features for prediction
    latest_features = features.iloc[-1].values.reshape(1, -1)
    
    prediction = model.predict(latest_features)[0]
    probability_of_up = model.predict_proba(latest_features)[0][1] # Probability of target being 1 (price up)

    return prediction, probability_of_up

