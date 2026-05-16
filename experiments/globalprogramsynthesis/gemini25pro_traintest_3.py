import skrub
import pandas as pd
import joblib
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

def train_model(df, labels):
    # TODO generate additional features
    # --- START OF INSERTED CODE ---

    # It's good practice to work on a copy to avoid side effects
    df_processed = df.copy()

    # 1. Date-based features from 'event_date'
    # Convert to datetime objects, coercing errors to NaT (Not a Time)
    df_processed['event_date'] = pd.to_datetime(df_processed['event_date'], errors='coerce')

    # Extract useful components
    df_processed['event_year'] = df_processed['event_date'].dt.year
    df_processed['event_month'] = df_processed['event_date'].dt.month
    df_processed['event_day_of_week'] = df_processed['event_date'].dt.dayofweek
    df_processed['event_is_weekend'] = df_processed['event_date'].dt.dayofweek >= 5

    # 2. Interaction features between numerical columns
    # This might capture a combined effect of score and quantity
    df_processed['score_x_quantity'] = df_processed['score'] * df_processed['quantity']

    # 3. Simple text-based features from 'misc'
    # The length of the text can sometimes be a useful signal
    df_processed['misc_length'] = df_processed['misc'].str.len().fillna(0)

    # 4. Clean up original columns that have been transformed or are not useful
    # 'ser' is likely a high-cardinality identifier, which is not useful for generalization
    # 'event_date' has been replaced by its components
    df_processed = df_processed.drop(columns=['ser', 'event_date'])

    # The dataframe passed to the vectorizer is the one with new features
    df = df_processed
    
    # --- END OF INSERTED CODE ---
    
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