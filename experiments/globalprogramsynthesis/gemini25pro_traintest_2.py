import skrub
import pandas as pd
import joblib
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier

def train_model(df, labels):
    #TODO generate additional features
    # Convert 'event_date' to datetime objects to extract temporal features
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')

    # Extract time-based features from the 'event_date'
    df['event_year'] = df['event_date'].dt.year
    df['event_month'] = df['event_date'].dt.month
    df['event_day'] = df['event_date'].dt.day
    df['event_dayofweek'] = df['event_date'].dt.dayofweek
    df['event_weekofyear'] = df['event_date'].dt.isocalendar().week.astype(int)
    
    # Create a simple interaction feature between two numerical columns
    # We'll fill potential division-by-zero or missing values with 0
    df['score_per_quantity'] = (df['score'] / df['quantity']).fillna(0)

    # Drop columns that are no longer needed or are not useful for the model
    # 'ser' is likely a high-cardinality identifier that we don't want the model to memorize.
    # The original 'event_date' has been replaced by more useful features.
    df = df.drop(columns=['ser', 'event_date'])
    
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