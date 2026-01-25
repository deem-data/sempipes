import shutil
from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

# Download latest version
path = kagglehub.dataset_download("mohammedalrashidan/contracts-clauses-datasets")

print("Path to dataset files:", path)

# Load the CSV file - try multiple possible locations and names
csv_path = None
for possible_path in [Path(path) / "legal_docs.csv"]:
    if possible_path.exists():
        csv_path = possible_path
        break

if csv_path is None:
    # Try to find any CSV file by searching
    csv_files = list(Path(path).rglob("*.csv"))
    if csv_files:
        csv_path = csv_files[0]
        print(f"Found CSV file: {csv_path}")

if csv_path is None or not csv_path.exists():
    raise FileNotFoundError(f"Could not find CSV file in {path}")

print(f"Found CSV at: {csv_path}")
# Read CSV with error handling
try:
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip", encoding="utf-8")
except TypeError:
    # Older pandas versions use error_bad_lines instead
    try:
        df = pd.read_csv(csv_path, engine="python", error_bad_lines=False, warn_bad_lines=True, encoding="utf-8")
    except TypeError:
        # Even older versions - use default engine
        df = pd.read_csv(csv_path, encoding="utf-8")
print(f"Loaded dataset with {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Take a stratified subsample based on clause_type
sample_size = min(10000, len(df))
df_subsample, _ = train_test_split(df, train_size=sample_size, stratify=df["clause_type"], random_state=42)
df_subsample = df_subsample.reset_index(drop=True)

print(f"Created stratified subsample with {len(df_subsample)} rows")
if "clause_type" in df_subsample.columns:
    print("Clause type distribution:")
    print(df_subsample["clause_type"].value_counts().sort_index())

# Create target directories
target_dir = Path(f"tests/data/contracts-dataset-{sample_size}")
target_dir.mkdir(parents=True, exist_ok=True)

# Save the subsampled CSV
output_path = target_dir / "contracts.csv"
df_subsample.to_csv(output_path, index=False)
print(f"Saved subsampled CSV to {output_path}")
print(f"Subsample contains {len(df_subsample)} rows")
