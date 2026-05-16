import skrub
import pandas as pd
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('experiments/globalprogramsynthesis/data.csv')

labels = df['label']
data = df.drop(columns=['label'])

#TODO: Add five cool extra features related to the mean, median and 90th percentile of the score column
# Calculate group-wise statistics based on categorical columns
mean_score_by_priority = data.groupby('priority')['score'].transform('mean')
median_score_by_status = data.groupby('status')['score'].transform('median')
p90_score_by_ser = data.groupby('ser')['score'].transform(lambda x: x.quantile(0.9))

# 1. Mean score for the item's priority level
data['mean_score_by_priority'] = mean_score_by_priority

# 2. Median score for the item's status
data['median_score_by_status'] = median_score_by_status

# 3. 90th percentile score for the item's series
data['p90_score_by_ser'] = p90_score_by_ser

# 4. Deviation of the score from the mean score of its priority group
data['score_deviation_from_priority_mean'] = data['score'] - mean_score_by_priority

# 5. Ratio of the score to the median score of its status group
# Adding a small epsilon to the denominator to avoid division by zero
data['score_ratio_to_status_median'] = data['score'] / (median_score_by_status + 1e-6)


train, test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vectorizer = TableVectorizer()
encoded_train = vectorizer.fit_transform(train)
model = HistGradientBoostingClassifier()
model.fit(encoded_train, y_train)

predictions = model.predict(vectorizer.transform(test))