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
    event_date=lambda df_: pd.to_datetime(df_['event_date'])
).assign(
    event_year=lambda df_: df_.event_date.dt.year,
    event_month=lambda df_: df_.event_date.dt.month,
    event_dayofweek=lambda df_: df_.event_date.dt.dayofweek,
    event_dayofyear=lambda df_: df_.event_date.dt.dayofyear,
    score_x_quantity=lambda df_: df_.score * df_.quantity,
    score_plus_extra=lambda df_: df_.score + df_.extra,
    ser_len=lambda df_: df_.ser.str.len(),
    misc_len=lambda df_: df_.misc.str.len(),
    score_per_quantity=lambda df_: df_.score / (df_.quantity + 1e-6),
    priority_flag_combo=lambda df_: df_.priority.astype(str) + "_" + df_.flag.astype(str)
)

encoded_data = with_features.skb.apply(TableVectorizer())

predictions = encoded_data.skb.apply(HistGradientBoostingClassifier(), y=labels)