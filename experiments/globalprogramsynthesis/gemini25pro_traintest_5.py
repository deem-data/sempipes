import skrub
import pandas as pd
import joblib
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier

def train_model(df, labels):
    #TODO generate additional features
    # Convert 'event_date' to datetime objects to extract temporal features
    df['event_date'] = pd.to_datetime(df['event_date'])

    # Extract date components which can be useful for the model
    df['event_year'] = df['event_date'].dt.year
    df['event_month'] = df['event_date'].dt.month
    df['event_day'] = df['event_date'].dt.day
    df['event_dayofweek'] = df['event_date'].dt.dayofweek

    # Create a simple interaction feature between two numerical columns
    df['score_x_quantity'] = df['score'] * df['quantity']

    # Drop columns that are less likely to be useful or have been replaced
    # 'ser' is likely a high-cardinality identifier that could cause overfitting
    # 'event_date' has been replaced by its components
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