import skrub
import pandas as pd
import joblib
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier

def train_model(df, labels):
    #TODO generate additional features
    # It's good practice to work on a copy to avoid side effects
    df = df.copy()

    # 1. Feature Engineering from 'event_date'
    # Convert string to datetime objects to enable feature extraction
    df['event_date'] = pd.to_datetime(df['event_date'])

    # Extract useful features like month, day of the week, etc.
    df['event_month'] = df['event_date'].dt.month
    df['event_day_of_week'] = df['event_date'].dt.dayofweek
    df['event_is_weekend'] = (df['event_date'].dt.dayofweek >= 5).astype(int)

    # Drop the original date column as we have extracted its information
    df = df.drop(columns=['event_date'])

    # 2. Feature Engineering from numerical columns
    # Create an interaction feature that might capture a combined effect
    df['score_quantity_interaction'] = df['score'] * df['quantity']
    
    # The TableVectorizer will automatically handle the remaining columns
    vectorizer = TableVectorizer()
    enc = vectorizer.fit_transform(df)
    model = HistGradientBoostingClassifier()
    model.fit(enc, labels)
    joblib.dump(vectorizer, "v.pkl")
    joblib.dump(model, "m.pkl")

def serve(data, path):
    encoder = joblib.load("v.pkl")
    predictor = joblib.load("m.pkl")
    return predictor.predict(encoder.transform(data))