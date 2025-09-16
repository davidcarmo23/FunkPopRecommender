import os
import pandas as pd

# Use project root as base
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
FILE_PATH = os.path.join(RAW_PATH, "funko.csv")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
CLEAN_FILE_PATH = os.path.join(PROCESSED_PATH, "funko_clean.csv")

FRANCHISE_MAP = {
    "Marvel": ["Spider-Man", "Iron Man", "Captain America"],
    "DC": ["Batman", "Superman", "Wonder Woman"],
    "Disney": ["Mickey", "Frozen", "Toy Story"],
    "Dragon Ball": ["Goku", "Vegeta", "Trunks", "Pan", "Bulma", "Goten", "Gohan"]
}

#extract franchise by mapping(ongoing)
def extract_franchise(name):
    name_lower = name.lower()
    for franchise, keywords in FRANCHISE_MAP.items():
        if any(keyword.lower() in name_lower for keyword in keywords):
            return franchise
    return name.split()[0]  # fallback to your method

def cleaner(df):
    # Remove duplicates
    df = df.drop_duplicates(subset=['name'])

    # Clean price data and normalize text
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])
    df['name'] = df['name'].str.strip().str.title()

    df['franchise'] = df['name'].apply(extract_franchise)

    # Validate required fields
    df = df.dropna(subset=['name', 'image', 'category'])

    return df

if __name__ == "__main__":
    cleaner()
