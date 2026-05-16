import skrub
import pandas as pd
import joblib
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier

def generate_features(df):
    """
    Generates additional features from the raw input data.
    This function is used in both training and serving to ensure consistency.
    """
    # Make a copy to avoid modifying the original DataFrame in place
    df_featured = df.copy()

    # Feature 1: Date-based features from 'event_date'
    # Convert column to datetime, coercing errors to NaT (Not a Time)
    df_featured['event_date'] = pd.to_datetime(df_featured['event_date'], errors='coerce')
    
    # Extract time-related features
    df_featured['event_year'] = df_featured['event_date'].dt.year
    df_featured['event_month'] = df_featured['event_date'].dt.month
    df_featured['event_dayofweek'] = df_featured['event_date'].dt.dayofweek
    df_featured['event_dayofyear'] = df_featured['event_date'].dt.dayofyear
    
    # Drop the original datetime column as its information is now captured
    df_featured = df_featured.drop(columns=['event_date'])

    # Feature 2: Categorical feature from 'ser' string
    # Assuming the prefix of 'ser' is a meaningful category
    df_featured['ser_prefix'] = df_featured['ser'].str[:3]

    # Feature 3: Interaction features between numerical columns
    df_featured['score_x_quantity'] = df_featured['score'] * df_featured['quantity']
    df_featured['score_x_extra'] = df_featured['score'] * df_featured['extra']

    # Feature 4: Length of the 'misc' text field
    df_featured['misc_len'] = df_featured['misc'].str.len()

    # Handle potential missing values that may have been introduced
    # (e.g., from failed date parsing or empty strings).
    # A simple fill with 0 is used here.
    df_featured = df_featured.fillna(0)
    
    return df_featured

def train_model(df, labels):
    # Generate additional features using the helper function
    df_featured = generate_features(df)
    
    vectorizer = TableVectorizer()
    enc = vectorizer.fit_transform(df_featured)
    model = HistGradientBoostingClassifier()
    model.fit(enc, labels)
    joblib.dump(vectorizer, "v.pkl")
    joblib.dump(model, "m.pkl")

def serve(data, path):
    encoder = joblib.load("v.pkl")
    predictor = joblib.load("m.pkl")

    # Apply the same feature engineering steps during inference
    data_featured = generate_features(data)
    
    return predictor.predict(encoder.transform(data_featured))