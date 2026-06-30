import pandas as pd
from pathlib import Path

# Folder containing all CSV files
DATA_FOLDER = Path("data")

# Find all CSV files
csv_files = list(DATA_FOLDER.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files.")

dfs = []

for file in csv_files:
    print(f"Loading: {file.name}")

    df = pd.read_csv(file)

    dfs.append(df)

# Merge all files
merged_df = pd.concat(
    dfs,
    ignore_index=True,
    sort=False
)

# Remove duplicate products
merged_df.drop_duplicates(
    subset=["product_id"],
    inplace=True
)

# Save merged dataset
output_file = DATA_FOLDER / "shopping_products.csv"

merged_df.to_csv(
    output_file,
    index=False
)

print("=" * 60)
print("Merged Successfully!")
print(f"Total Products : {len(merged_df)}")
print(f"Saved To : {output_file}")
print("=" * 60)