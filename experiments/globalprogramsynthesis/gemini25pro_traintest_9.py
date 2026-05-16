import skrub
import pandas as pd
import joblib
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

def _create_features(df):
    """Helper function to generate features for both training and serving."""
    # Make a copy to avoid modifying the original DataFrame via reference
    df = df.copy()

    # Convert 'event_date' from string to datetime objects
    df['event_date'] = pd.to_datetime(df['event_date'])

    # Extract time-based features from the date
    df['event_year'] = df['event_date'].dt.year
    df['event_month'] = df['event_date'].dt.month
    df['event_dayofweek'] = df['event_date'].dt.dayofweek
    df['event_weekofyear'] = df['event_date'].dt.isocalendar().week.astype(int)


    # Create an interaction feature between 'score' and 'quantity'
    df['score_quantity_interaction'] = df['score'] * df['quantity']

    # Create a simple feature from the 'misc' column, like its length
    df['misc_length'] = df['misc'].str.len()

    # Drop columns that are no longer needed or are potentially noisy identifiers
    # 'ser' is likely a unique ID with no predictive power.
    # 'event_date' has been replaced by more useful features.
    df = df.drop(columns=['ser', 'event_date'])
    
    return df


def train_model(df, labels):
    # Generate additional features
    df_featured = _create_features(df)

    # The TableVectorizer will automatically handle the different data types
    # of the original and newly created columns (numerical, categorical).
    vectorizer = TableVectorizer()
    enc = vectorizer.fit_transform(df_featured)
    
    model = HistGradientBoostingClassifier()
    model.fit(enc, labels)
    
    # Save the fitted vectorizer and model for serving
    joblib.dump(vectorizer, "v.pkl")
    joblib.dump(model, "m.pkl")

def serve(data, path):
    # Load the pre-trained encoder and predictor
    encoder = joblib.load("v.pkl")
    predictor = joblib.load("m.pkl")

    # IMPORTANT: Apply the same feature engineering steps to the incoming data
    data_featured = _create_features(data)
    
    # Transform the new data using the fitted encoder and make predictions
    return predictor.predict(encoder.transform(data_featured))