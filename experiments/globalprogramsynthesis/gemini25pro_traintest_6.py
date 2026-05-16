import skrub
import pandas as pd
import joblib
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier

def train_model(df, labels):
    #TODO generate additional features
    # --- Feature Engineering ---
    # Make a copy to avoid potential SettingWithCopyWarning
    df = df.copy()

    # 1. Date-based features
    # Convert 'event_date' to datetime, coercing errors to NaT (Not a Time)
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')

    # Extract date components. NaT values will result in NaN.
    df['event_year'] = df['event_date'].dt.year
    df['event_month'] = df['event_date'].dt.month
    df['event_day'] = df['event_date'].dt.day
    df['event_dayofweek'] = df['event_date'].dt.dayofweek

    # 2. Interaction feature
    # Combine score and quantity to create a new metric
    df['score_x_quantity'] = df['score'] * df['quantity']

    # 3. Text-based features
    # Get the length of string columns as a proxy for information content
    df['ser_len'] = df['ser'].str.len()
    df['misc_len'] = df['misc'].str.len()
    
    # 4. Clean up
    # Fill any NaNs created during feature engineering (e.g., from invalid dates)
    df.fillna(0, inplace=True)
    # --- End of Feature Engineering ---

    vectorizer = TableVectorizer()
    enc = vectorizer.fit_transform(df)
    model = HistGradientBoostingClassifier()
    model.fit(enc, labels)
    joblib.dump(vectorizer, "v.pkl")
    joblib.dump(model, "m.pkl")

def serve(data, path):
    encoder = joblib.load("v.pkl")
    predictor = joblib.load("m.pkl")

    # --- Feature Engineering (must be identical to training) ---
    # Make a copy to avoid modifying original data
    data = data.copy()

    # 1. Date-based features
    data['event_date'] = pd.to_datetime(data['event_date'], errors='coerce')
    data['event_year'] = data['event_date'].dt.year
    data['event_month'] = data['event_date'].dt.month
    data['event_day'] = data['event_date'].dt.day
    data['event_dayofweek'] = data['event_date'].dt.dayofweek

    # 2. Interaction feature
    data['score_x_quantity'] = data['score'] * data['quantity']

    # 3. Text-based features
    data['ser_len'] = data['ser'].str.len()
    data['misc_len'] = data['misc'].str.len()
    
    # 4. Clean up
    data.fillna(0, inplace=True)
    # --- End of Feature Engineering ---

    return predictor.predict(encoder.transform(data))