import skrub
import pandas as pd
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('experiments/globalprogramsynthesis/data.csv')

labels = df['label']
data = df.drop(columns=['label'])

#TODO: Add five cool extra features related to the mean, median and 90th percentile of the score column
# Feature 1: Deviation of each score from the global mean score
data['score_deviation_from_mean'] = data['score'] - data['score'].mean()

# Feature 2: Mean score grouped by 'priority'
# This captures the average score for each priority level.
data['mean_score_by_priority'] = data.groupby('priority')['score'].transform('mean')

# Feature 3: Median score grouped by 'status'
# This captures the typical score for each status, robust to outliers.
data['median_score_by_status'] = data.groupby('status')['score'].transform('median')

# Feature 4: 90th percentile score grouped by 'ser'
# This captures the high-end score distribution for each 'ser' category.
data['p90_score_by_ser'] = data.groupby('ser')['score'].transform(lambda x: x.quantile(0.9))

# Feature 5: Score as a ratio of the median score for its priority group.
# This normalizes the score within its priority category.
median_by_priority = data.groupby('priority')['score'].transform('median')
# Add a small epsilon to the denominator to prevent division by zero
data['score_ratio_to_median_by_priority'] = data['score'] / (median_by_priority + 1e-6)

train, test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vectorizer = TableVectorizer()
encoded_train = vectorizer.fit_transform(train)
model = HistGradientBoostingClassifier()
model.fit(encoded_train, y_train)

predictions = model.predict(vectorizer.transform(test))