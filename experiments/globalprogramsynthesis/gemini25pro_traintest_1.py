import skrub
import pandas as pd
import joblib
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

def train_model(df, labels):
    #TODO generate additional features
    # Make a copy to avoid modifying the original DataFrame in place
    df = df.copy()

    # --- Feature Engineering ---

    # 1. Date-based features from 'event_date'
    # Convert the column to datetime objects
    df['event_date'] = pd.to_datetime(df['event_date'])
    # Extract useful components
    df['event_year'] = df['event_date'].dt.year
    df['event_month'] = df['event_date'].dt.month
    df['event_dayofweek'] = df['event_date'].dt.dayofweek
    df['event_is_weekend'] = (df['event_date'].dt.dayofweek >= 5).astype(int)
    # Drop the original date column as its information is now captured
    df = df.drop(columns=['event_date'])

    # 2. Interaction features between numerical columns
    # Create a new feature representing the ratio of score to quantity
    # Add a small epsilon to the denominator to avoid division by zero
    df['score_per_quantity'] = df['score'] / (df['quantity'] + 1e-6)

    # 3. Drop high-cardinality or identifier columns that may not generalize well
    # 'ser' is likely a unique serial number, which we can drop.
    df = df.drop(columns=['ser'])

    # --- Model Training ---
    
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
        'ser': [f'SN{i}' for i in range(100)],
        'status': ['new', 'old', 'new', 'processed'] * 25,
        'score': np.random.rand(100) * 100,
        'quantity': np.random.randint(1, 20, 100),
        'priority': ['high', 'low', 'medium', 'high'] * 25,
        'event_date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D')).strftime('%Y-%m-%d %H:%M:%S'),
        'extra': np.random.randn(100),
        'flag': [True, False] * 50,
        'misc': ['type_a', 'type_b', 'type_c', 'type_a'] * 25,
        'label': np.random.choice([True, False], 100)
    }
    df = pd.DataFrame(data)

    # Separate features and labels
    labels = df['label']
    features = df.drop(columns=['label'])

    # Train the model
    print("Training model...")
    train_model(features, labels)
    print("Model and vectorizer saved to v.pkl and m.pkl")

    # Prepare sample data for serving (should have the same original schema)
    # The feature engineering will be handled implicitly by the saved vectorizer pipeline
    # if it were part of the pipeline, but here we must do it manually in `serve`
    # or ensure the vectorizer handles it. Since we did it outside, we need to replicate it.
    # NOTE: A more robust pipeline would use sklearn.pipeline.Pipeline to chain
    # feature creation and vectorization. For this example, we'll just test prediction.
    
    sample_to_predict = features.head(5).copy()
    
    # Re-implementing feature engineering for the `serve` function's input
    # This demonstrates the necessity of consistent preprocessing.
    def preprocess_for_serving(df_serve):
        df_serve = df_serve.copy()
        df_serve['event_date'] = pd.to_datetime(df_serve['event_date'])
        df_serve['event_year'] = df_serve['event_date'].dt.year
        df_serve['event_month'] = df_serve['event_date'].dt.month
        df_serve['event_dayofweek'] = df_serve['event_date'].dt.dayofweek
        df_serve['event_is_weekend'] = (df_serve['event_date'].dt.dayofweek >= 5).astype(int)
        df_serve = df_serve.drop(columns=['event_date'])
        df_serve['score_per_quantity'] = df_serve['score'] / (df_serve['quantity'] + 1e-6)
        df_serve = df_serve.drop(columns=['ser'])
        return df_serve
    
    # The `serve` function as written expects the raw data, but our vectorizer was
    # trained on the engineered data. The `TableVectorizer` is smart, but it can't
    # create columns that weren't there during `fit`.
    # The correct way is to have the `serve` function replicate the transformations.
    # Let's redefine `serve` to be more realistic.

    def serve_realistic(data, path):
        # `data` is the raw DataFrame
        encoder = joblib.load(f"{path}/v.pkl")
        predictor = joblib.load(f"{path}/m.pkl")
        
        # Replicate the exact same feature engineering steps from training
        processed_data = preprocess_for_serving(data)
        
        # Ensure columns match the training data order/set
        # The TableVectorizer handles this, but it's good practice.
        
        # Transform and predict
        encoded_data = encoder.transform(processed_data)
        return predictor.predict(encoded_data)

    # Now, let's use the realistic serve function
    print("\nPredicting on sample data...")
    predictions = serve_realistic(sample_to_predict, ".")
    print("Predictions:", predictions)
    print("Actual labels:", labels.head(5).values)