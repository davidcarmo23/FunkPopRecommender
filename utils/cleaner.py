import os
import pandas as pd

# Use project root as base
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
FILE_PATH = os.path.join(RAW_PATH, "funko.csv")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
CLEAN_FILE_PATH = os.path.join(PROCESSED_PATH, "funko_clean.csv")

def cleaner():
    # Ensure folders exist
    os.makedirs(RAW_PATH, exist_ok=True)
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    # Load raw CSV
    df = pd.read_csv(FILE_PATH)
    print(df["price"])

    # Add franchise column
    df["franchise"] = df["name"].apply(lambda x: x.split()[0])

    # Save cleaned CSV
    df.to_csv(CLEAN_FILE_PATH, index=False)
    print(f"Cleaned data saved to {CLEAN_FILE_PATH}")

if __name__ == "__main__":
    cleaner()
