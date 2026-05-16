import skrub
import pandas as pd
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('experiments/globalprogramsynthesis/data.csv')

labels = df['label']
data = df.drop(columns=['label'])

#TODO: Add five cool extra features related to the mean, median and 90th percentile of the score column
# Calculate aggregate statistics for the 'score' column
mean_score = data['score'].mean()
median_score = data['score'].median()
p90_score = data['score'].quantile(0.9)

# Feature 1: Deviation from the mean score
data['score_deviation_from_mean'] = data['score'] - mean_score

# Feature 2: Deviation from the median score
data['score_deviation_from_median'] = data['score'] - median_score

# Feature 3: Boolean flag if score is above the mean
data['is_above_mean'] = (data['score'] > mean_score).astype(int)

# Feature 4: Boolean flag if score is above the median
data['is_above_median'] = (data['score'] > median_score).astype(int)

# Feature 5: Boolean flag if score is in the top 10th percentile
data['is_top_10_percentile'] = (data['score'] > p90_score).astype(int)


train, test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vectorizer = TableVectorizer()
encoded_train = vectorizer.fit_transform(train)
model = HistGradientBoostingClassifier()
model.fit(encoded_train, y_train)

predictions = model.predict(vectorizer.transform(test))