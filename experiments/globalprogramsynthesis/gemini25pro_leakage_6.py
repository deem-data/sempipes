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
score_p90 = data['score'].quantile(0.9)

# Feature 1: Difference between the score and the overall mean score
data['score_minus_mean'] = data['score'] - score_mean

# Feature 2: Difference between the score and the overall median score
data['score_minus_median'] = data['score'] - score_median

# Feature 3: A binary flag indicating if the score is in the top 10%
data['is_above_p90'] = (data['score'] > score_p90).astype(int)

# Feature 4: The ratio of the score to the mean score
# (adding a small epsilon to avoid potential division by zero)
data['score_to_mean_ratio'] = data['score'] / (score_mean + 1e-9)

# Feature 5: The ratio of the score to the median score
# (adding a small epsilon to avoid potential division by zero)
data['score_to_median_ratio'] = data['score'] / (score_median + 1e-9)


train, test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vectorizer = TableVectorizer()
encoded_train = vectorizer.fit_transform(train)
model = HistGradientBoostingClassifier()
model.fit(encoded_train, y_train)

predictions = model.predict(vectorizer.transform(test))