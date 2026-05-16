import skrub
import pandas as pd
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('experiments/globalprogramsynthesis/data.csv')

labels = df['label']
data = df.drop(columns=['label'])

#TODO: Add five cool extra features related to the mean, median and 90th percentile of the score column
# Calculate global statistics for the 'score' column
mean_score = data['score'].mean()
median_score = data['score'].median()
p90_score = data['score'].quantile(0.9)

# 1. Deviation from the global mean score
data['score_dev_from_mean'] = data['score'] - mean_score

# 2. Deviation from the global median score
data['score_dev_from_median'] = data['score'] - median_score

# 3. Binary flag indicating if the score is in the top 10%
data['is_in_top_10_percentile'] = (data['score'] > p90_score).astype(int)

# 4. Mean score calculated per 'priority' category
# This feature provides context about the typical score for a given priority level.
data['mean_score_by_priority'] = data.groupby('priority')['score'].transform('mean')

# 5. Score deviation from the mean score within its own 'priority' group
# This normalizes the score based on its priority context.
data['score_dev_from_priority_mean'] = data['score'] - data['mean_score_by_priority']


train, test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vectorizer = TableVectorizer()
encoded_train = vectorizer.fit_transform(train)
model = HistGradientBoostingClassifier()
model.fit(encoded_train, y_train)

predictions = model.predict(vectorizer.transform(test))