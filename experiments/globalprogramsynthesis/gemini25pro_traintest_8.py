import skrub
import pandas as pd
import joblib
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier

def train_model(df, labels):
    #TODO generate additional features
    # Make a copy to avoid modifying the original DataFrame passed to the function
    df = df.copy()

    # Drop the 'ser' column as it's likely an uninformative identifier
    if 'ser' in df.columns:
        df = df.drop(columns=['ser'])

    # Feature Engineering for 'event_date'
    # Convert 'event_date' string to datetime objects to extract features
    df['event_date'] = pd.to_datetime(df['event_date'])

    # Extract time-based features
    df['event_year'] = df['event_date'].dt.year
    df['event_month'] = df['event_date'].dt.month
    df['event_day'] = df['event_date'].dt.day
    df['event_dayofweek'] = df['event_date'].dt.dayofweek  # Monday=0, Sunday=6
    df['event_is_weekend'] = (df['event_date'].dt.dayofweek >= 5).astype(int)

    # Drop the original date column as its information is now captured in other features
    df = df.drop(columns=['event_date'])

    # Create an interaction feature between 'score' and 'quantity'
    if 'score' in df.columns and 'quantity' in df.columns:
        df['total_value'] = df['score'] * df['quantity']
    
    # Convert boolean 'flag' to integer (0 or 1)
    if 'flag' in df.columns:
        df['flag'] = df['flag'].astype(int)

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