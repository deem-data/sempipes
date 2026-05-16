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

# 1. Difference between the score and the mean score
data['score_mean_diff'] = data['score'] - score_mean

# 2. Difference between the score and the median score
data['score_median_diff'] = data['score'] - score_median

# 3. Difference between the score and the 90th percentile score
data['score_p90_diff'] = data['score'] - score_p90

# 4. Binary feature indicating if the score is above the mean
data['is_above_mean'] = (data['score'] > score_mean).astype(int)

# 5. Binary feature indicating if the score is a "high score" (above 90th percentile)
data['is_high_score'] = (data['score'] > score_p90).astype(int)


train, test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vectorizer = TableVectorizer()
encoded_train = vectorizer.fit_transform(train)
model = HistGradientBoostingClassifier()
model.fit(encoded_train, y_train)

predictions = model.predict(vectorizer.transform(test))