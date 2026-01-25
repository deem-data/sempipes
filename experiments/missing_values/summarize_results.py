import json
import pandas as pd
import numpy as np
from pathlib import Path

# Read the JSON file
results_file = Path(__file__).parent / "cancer_5runs_gpt5.json"
results = []

with open(results_file, 'r') as f:
    for line in f:
        if line.strip():
            try:
                data = json.loads(line)
                # Convert NaN string to actual NaN
                if isinstance(data.get('mse'), str) and data['mse'].lower() == 'nan':
                    data['mse'] = np.nan
                results.append(data)
            except json.JSONDecodeError:
                continue

# Convert to DataFrame
df = pd.DataFrame(results)

# Convert mse to numeric, handling NaN
df['mse'] = pd.to_numeric(df['mse'], errors='coerce')

# Group by dataset, missingness, and imputer, then calculate mean and std
summary = df.groupby(['data', 'missingness', 'imputer'])['mse'].agg(['mean', 'std', 'count']).reset_index()

# Print results table
print("=" * 120)
print("BENCHMARK RESULTS SUMMARY")
print("=" * 120)
print()

# Group by dataset
for dataset in sorted(df['data'].unique()):
    print("=" * 120)
    print(f"DATASET: {dataset}")
    print("=" * 120)
    print()
    
    dataset_df = summary[summary['data'] == dataset]
    
    for missingness in ['MCAR', 'MAR', 'MNAR']:
        print("-" * 120)
        print(f"Missingness: {missingness}")
        print("-" * 120)
        print(f"{'Imputer':<30} {'Mean MSE':<20} {'Std MSE':<20} {'Count':<10}")
        print("-" * 120)
        
        missingness_df = dataset_df[dataset_df['missingness'] == missingness].sort_values('mean', na_position='last')
        
        if len(missingness_df) == 0:
            print("No results available")
        else:
            for _, row in missingness_df.iterrows():
                mean_str = f"{row['mean']:.6f}" if pd.notna(row['mean']) else "NaN"
                std_str = f"{row['std']:.6f}" if pd.notna(row['std']) else "NaN"
                count_str = str(int(row['count']))
                print(f"{row['imputer']:<30} {mean_str:<20} {std_str:<20} {count_str:<10}")
        
        print("-" * 120)
        print()

print("=" * 120)
