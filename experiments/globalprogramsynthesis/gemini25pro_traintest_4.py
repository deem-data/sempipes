import skrub
import pandas as pd
import joblib
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier

def train_model(df, labels):
    #TODO generate additional features
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Drop the 'ser' column as it's likely a unique identifier with no predictive power
    if 'ser' in df.columns:
        df = df.drop(columns=['ser'])

    # Feature Engineering for 'event_date'
    # Convert to datetime objects
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Extract time-based features
    df['event_year'] = df['event_date'].dt.year
    df['event_month'] = df['event_date'].dt.month
    df['event_day_of_week'] = df['event_date'].dt.dayofweek
    df['event_week_of_year'] = df['event_date'].dt.isocalendar().week.astype(int)
    
    # Drop the original datetime column as its information is now captured
    df = df.drop(columns=['event_date'])

    # Create an interaction feature
    # This could represent a total value or importance
    df['score_quantity_interaction'] = df['score'] * df['quantity']
    
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

# Example Usage (for demonstration purposes)
if __name__ == '__main__':
    # Create a sample DataFrame based on the schema
    data = {
        'ser': [f'ID_{i}' for i in range(100)],
        'status': ['new', 'old', 'new', 'pending'] * 25,
        'score': [i * 0.5 for i in range(100)],
        'quantity': [i % 10 + 1 for i in range(100)],
        'priority': ['high', 'low', 'medium', 'high', 'low'] * 20,
        'event_date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D')).strftime('%Y-%m-%d'),
        'extra': [i / 10.0 for i in range(100)],
        'flag': [True, False] * 50,
        'misc': ['category_a', 'category_b', 'category_c', 'category_a', 'unknown'] * 20,
        'label': [True if i % 3 == 0 else False for i in range(100)]
    }
    sample_df = pd.DataFrame(data)
    
    # Separate features and labels
    X = sample_df.drop(columns=['label'])
    y = sample_df['label']
    
    # Train the model
    print("Training model...")
    train_model(X, y)
    print("Model and vectorizer saved to v.pkl and m.pkl")
    
    # Prepare some new data for serving/prediction
    new_data_dict = {
        'ser': ['ID_101', 'ID_102'],
        'status': ['new', 'pending'],
        'score': [55.5, 23.0],
        'quantity': [5, 2],
        'priority': ['high', 'low'],
        'event_date': ['2024-08-15', '2024-08-16'],
        'extra': [10.1, 2.3],
        'flag': [False, True],
        'misc': ['category_b', 'new_category']
    }
    new_data_df = pd.DataFrame(new_data_dict)
    
    # Serve the model
    print("\nMaking predictions on new data...")
    predictions = serve(new_data_df, path='.')
    print(f"Predictions: {predictions}")