import pandas as pd

data = pd.read_csv("experiments/aide_failures_fairness/repayments.csv")

# Create gender grouping (Male vs non-Male)
data["gender_group"] = data["gender"].apply(lambda x: "Male" if x == "M" else "Non-Male")

# Compute mean statistics by gender group
stats_by_gender = (
    data.groupby("gender_group")
    .agg({"income": "mean", "dti": "mean", "credit_score": "mean", "repaid": "mean"})
    .round(4)
)


sample_sizes = data["gender_group"].value_counts()
stats_by_gender["sample_size"] = sample_sizes

# Display results
print("Mean Statistics by Gender Group:")
print("=" * 50)
print(stats_by_gender)
