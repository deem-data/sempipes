import skrub
import pandas as pd
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('experiments/globalprogramsynthesis/data.csv')

labels = df['label']
data = df.drop(columns=['label'])

#TODO: Add five cool extra features related to the mean, median and 90th percentile of the score column
# Calculate the required statistics from the 'score' column
score_mean = data['score'].mean()
score_median = data['score'].median()
score_p90 = data['score'].quantile(0.9)
score_std = data['score'].std()

# 1. Deviation from the mean score
data['score_deviation_from_mean'] = data['score'] - score_mean

# 2. Deviation from the median score (robust to outliers)
data['score_deviation_from_median'] = data['score'] - score_median

# 3. Binary flag for scores in the top 10%
data['is_in_top_10_percentile'] = (data['score'] > score_p90).astype(int)

# 4. Standardized score (Z-score)
# Add a small epsilon to avoid division by zero if std is 0
data['score_z_score'] = (data['score'] - score_mean) / (score_std + 1e-9)

# 5. Score as a ratio of the median
# Add a small epsilon to avoid division by zero if median is 0
data['score_ratio_to_median'] = data['score'] / (score_median + 1e-9)


train, test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vectorizer = TableVectorizer()
encoded_train = vectorizer.fit_transform(train)
model = HistGradientBoostingClassifier()
model.fit(encoded_train, y_train)

predictions = model.predict(vectorizer.transform(test))