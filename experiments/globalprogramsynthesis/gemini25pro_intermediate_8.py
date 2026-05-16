import skrub
import pandas as pd
from skrub import DropUninformative, TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv('experiments/globalprogramsynthesis/data.csv')
all_data = skrub.var("all_data", df)

labels = all_data['label'].skb.mark_as_y()
data = all_data.drop(columns=['label']).skb.mark_as_X()

dropper = DropUninformative(drop_if_constant=True, drop_if_unique=True, drop_null_fraction=0.9)
cleaned_data = data.skb.apply(dropper)

#TODO: Add ten cool extra features derived from existing columns
with_features = cleaned_data.assign(
    # Create an intermediate datetime column for feature engineering
    event_date_dt=lambda df: pd.to_datetime(df['event_date'])
).assign(
    # 1. Extract month from date
    event_month=lambda df: df['event_date_dt'].dt.month,
    # 2. Extract day of the week
    event_dayofweek=lambda df: df['event_date_dt'].dt.dayofweek,
    # 3. Check if the event is on a weekend
    is_weekend=lambda df: df['event_dayofweek'].isin([5, 6]),
    # 4. Calculate a ratio of two numeric features
    score_per_extra=lambda df: df['score'] / (df['extra'] + 1e-6),
    # 5. Create an interaction term between score and quantity
    score_x_quantity=lambda df: df['score'] * df['quantity'],
    # 6. Create a polynomial feature
    score_squared=lambda df: df['score'] ** 2,
    # 7. Combine two categorical features into one
    priority_status=lambda df: df['priority'].astype(str) + "_" + df['status'].astype(str),
    # 8. Calculate the length of a text field
    misc_len=lambda df: df['misc'].str.len().fillna(0).astype(int),
    # 9. Interaction between a numeric and a boolean feature
    score_if_flag=lambda df: df['score'] * df['flag'].astype(int),
    # 10. Time-based feature: days since the first event in the dataset
    days_since_start=lambda df: (df['event_date_dt'] - df['event_date_dt'].min()).dt.days
).drop(columns=['event_date_dt']) # Drop the intermediate datetime column

encoded_data = with_features.skb.apply(TableVectorizer())

predictions = encoded_data.skb.apply(HistGradientBoostingClassifier(), y=labels)