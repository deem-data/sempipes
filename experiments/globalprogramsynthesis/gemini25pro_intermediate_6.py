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
    cleaned_data.assign(event_date=lambda df: pd.to_datetime(df['event_date']))
                .assign(
                    # 1. Year from event_date
                    event_year=lambda df: df['event_date'].dt.year,
                    # 2. Month from event_date
                    event_month=lambda df: df['event_date'].dt.month,
                    # 3. Day of the week from event_date
                    day_of_week=lambda df: df['event_date'].dt.dayofweek,
                    # 4. Week of the year from event_date
                    week_of_year=lambda df: df['event_date'].dt.isocalendar().week.astype(int),
                    # 5. Interaction between score and quantity
                    score_x_quantity=lambda df: df['score'] * df['quantity'],
                    # 6. Ratio of score to quantity (with epsilon for stability)
                    score_per_quantity=lambda df: df['score'] / (df['quantity'] + 1e-6),
                    # 7. Length of the 'ser' string
                    ser_length=lambda df: df['ser'].str.len(),
                    # 8. Interaction between two categorical features
                    status_priority_combo=lambda df: df['status'].astype(str) + '_' + df['priority'].astype(str),
                    # 9. A boolean feature indicating if priority is 'high'
                    is_high_priority=lambda df: (df['priority'] == 'high').astype(int),
                    # 10. A combined numerical feature
                    score_plus_extra=lambda df: df['score'] + df['extra']
                )
)

encoded_data = with_features.skb.apply(TableVectorizer())

predictions = encoded_data.skb.apply(HistGradientBoostingClassifier(), y=labels)