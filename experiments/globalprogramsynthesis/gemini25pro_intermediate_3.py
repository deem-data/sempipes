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
    event_date=pd.to_datetime(cleaned_data['event_date'])
).assign(
    # 1. Day of the week from event_date
    event_day_of_week=lambda df_: df_['event_date'].dt.dayofweek,
    # 2. Month from event_date
    event_month=lambda df_: df_['event_date'].dt.month,
    # 3. Boolean for weekend
    is_weekend=lambda df_: df_['event_date'].dt.dayofweek.isin([5, 6]),
    # 4. Interaction between score and quantity
    score_x_quantity=lambda df_: df_['score'] * df_['quantity'],
    # 5. Ratio of score to extra, with a small epsilon to avoid division by zero
    score_per_extra=lambda df_: df_['score'] / (df_['extra'] + 1e-6),
    # 6. Length of the 'ser' string identifier
    ser_len=lambda df_: df_['ser'].str.len(),
    # 7. Check if 'misc' string contains any digit
    misc_has_digit=lambda df_: df_['misc'].str.contains(r'\d', na=False),
    # 8. Ordinal encoding for priority
    priority_numeric=lambda df_: df_['priority'].map({'low': 0, 'medium': 1, 'high': 2}),
    # 9. Interaction between two categorical features
    status_priority_interaction=lambda df_: df_['status'] + "_" + df_['priority'],
    # 10. Convert boolean flag to integer
    flag_as_int=lambda df_: df_['flag'].astype(int)
)

encoded_data = with_features.skb.apply(TableVectorizer())

predictions = encoded_data.skb.apply(HistGradientBoostingClassifier(), y=labels)