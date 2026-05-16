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
mean_score = data['score'].mean()
median_score = data['score'].median()
p90_score = data['score'].quantile(0.9)

# Feature 1: Difference between the score and the mean score
data['score_minus_mean'] = data['score'] - mean_score

# Feature 2: Difference between the score and the median score
data['score_minus_median'] = data['score'] - median_score

# Feature 3: Boolean flag indicating if the score is in the top 10%
data['is_top_10_percent_score'] = data['score'] > p90_score

# Feature 4: Ratio of the score to the median score
# Adding a small epsilon to avoid division by zero
data['score_to_median_ratio'] = data['score'] / (median_score + 1e-6)

# Feature 5: Categorical feature binning scores based on quantiles
def assign_score_bin(score, median, p90):
    if score > p90:
        return 'High'
    elif score > median:
        return 'Medium'
    else:
        return 'Low'

data['score_bin'] = data['score'].apply(lambda x: assign_score_bin(x, median_score, p90_score))


train, test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vectorizer = TableVectorizer()
encoded_train = vectorizer.fit_transform(train)
model = HistGradientBoostingClassifier()
model.fit(encoded_train, y_train)

predictions = model.predict(vectorizer.transform(test))