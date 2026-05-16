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
score_mean = data['score'].mean()
score_median = data['score'].median()
score_90th_percentile = data['score'].quantile(0.9)

# Feature 1: Difference from the global mean score
data['score_diff_from_mean'] = data['score'] - score_mean

# Feature 2: Difference from the global median score
data['score_diff_from_median'] = data['score'] - score_median

# Feature 3: Binary flag for scores above the 90th percentile (potential outliers)
data['is_score_in_top_10_percent'] = (data['score'] > score_90th_percentile).astype(int)

# Feature 4: Score relative to the mean score for its 'priority' group
priority_mean_score = data.groupby('priority')['score'].transform('mean')
data['score_vs_priority_group_mean'] = data['score'] - priority_mean_score

# Feature 5: Score relative to the median score for its 'status' group
status_median_score = data.groupby('status')['score'].transform('median')
data['score_vs_status_group_median'] = data['score'] - status_median_score


train, test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vectorizer = TableVectorizer()
encoded_train = vectorizer.fit_transform(train)
model = HistGradientBoostingClassifier()
model.fit(encoded_train, y_train)

predictions = model.predict(vectorizer.transform(test))