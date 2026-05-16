import skrub
import pandas as pd
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('experiments/globalprogramsynthesis/data.csv')

labels = df['label']
data = df.drop(columns=['label'])

#TODO: Add five cool extra features related to the mean, median and 90th percentile of the score column
# Calculate key statistics for the 'score' column
score_mean = data['score'].mean()
score_median = data['score'].median()
score_p90 = data['score'].quantile(0.9)

# 1. Feature: Deviation of the score from the mean
# This centers the score, showing how much it differs from the average.
data['score_dev_from_mean'] = data['score'] - score_mean

# 2. Feature: Deviation of the score from the median
# Similar to the mean deviation, but more robust to outliers.
data['score_dev_from_median'] = data['score'] - score_median

# 3. Feature: Binary indicator for score being above the mean
# A simple flag to capture whether a score is in the upper half of the distribution.
data['is_above_mean'] = (data['score'] > score_mean).astype(int)

# 4. Feature: Binary indicator for score being in the top 10%
# This flags high-performing scores, which could be a strong signal.
data['is_top_10_percent'] = (data['score'] > score_p90).astype(int)

# 5. Feature: Score as a ratio of the 90th percentile
# Normalizes the score relative to a high benchmark, showing its relative strength.
# Added a small epsilon to prevent division by zero.
data['score_ratio_to_p90'] = data['score'] / (score_p90 + 1e-6)


train, test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vectorizer = TableVectorizer()
encoded_train = vectorizer.fit_transform(train)
model = HistGradientBoostingClassifier()
model.fit(encoded_train, y_train)

predictions = model.predict(vectorizer.transform(test))