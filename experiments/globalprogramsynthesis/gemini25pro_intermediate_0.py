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
    event_date=lambda df: pd.to_datetime(df['event_date'])
).assign(
    event_year=lambda df: df['event_date'].dt.year,
    event_month=lambda df: df['event_date'].dt.month,
    event_day_of_week=lambda df: df['event_date'].dt.dayofweek,
    is_weekend=lambda df: df['event_date'].dt.dayofweek >= 5,
    score_x_quantity=lambda df: df['score'] * df['quantity'],
    score_div_extra=lambda df: df['score'] / (df['extra'] + 1e-6),
    quantity_x_extra=lambda df: df['quantity'] * df['extra'],
    ser_length=lambda df: df['ser'].str.len(),
    misc_has_digit=lambda df: df['misc'].str.contains(r'\d', na=False),
    priority_is_high=lambda df: df['priority'] == 'high'
)

encoded_data = with_features.skb.apply(TableVectorizer())

predictions = encoded_data.skb.apply(HistGradientBoostingClassifier(), y=labels)