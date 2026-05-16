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
    event_date=lambda x: pd.to_datetime(x['event_date'])
).assign(
    # Date-based features
    event_dayofweek=lambda x: x['event_date'].dt.dayofweek,
    event_month=lambda x: x['event_date'].dt.month,
    event_is_weekend=lambda x: x['event_date'].dt.dayofweek.isin([5, 6]),
    # Numerical interaction features
    score_x_quantity=lambda x: x['score'] * x['quantity'],
    score_per_extra=lambda x: x['score'] / (x['extra'] + 1e-6),
    quantity_log=lambda x: (x['quantity'] + 1).log(),
    # Categorical and boolean transformations
    priority_numeric=lambda x: x['priority'].map({'low': 0, 'medium': 1, 'high': 2}),
    flag_as_int=lambda x: x['flag'].astype(int),
    # String-based features
    ser_prefix=lambda x: x['ser'].str[:2],
    status_priority_interaction=lambda x: x['status'] + '_' + x['priority']
)

encoded_data = with_features.skb.apply(TableVectorizer())

predictions = encoded_data.skb.apply(HistGradientBoostingClassifier(), y=labels)