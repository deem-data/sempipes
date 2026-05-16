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
with_features = (
    cleaned_data
    .assign(event_date_dt=lambda df: pd.to_datetime(df['event_date']))
    .assign(
        event_year=lambda df: df['event_date_dt'].dt.year,
        event_month=lambda df: df['event_date_dt'].dt.month,
        event_day_of_week=lambda df: df['event_date_dt'].dt.dayofweek,
        is_weekend=lambda df: df['event_date_dt'].dt.dayofweek.isin([5, 6]).astype(int),
        score_x_quantity=lambda df: df['score'] * df['quantity'],
        score_div_extra=lambda df: df['score'] / (df['extra'] + 1e-6),
        ser_length=lambda df: df['ser'].str.len(),
        misc_length=lambda df: df['misc'].str.len(),
        priority_numeric=lambda df: df['priority'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(0),
        score_squared=lambda df: df['score'] ** 2
    )
)

encoded_data = with_features.skb.apply(TableVectorizer())

predictions = encoded_data.skb.apply(HistGradientBoostingClassifier(), y=labels)