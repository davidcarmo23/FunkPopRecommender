import os
import json
import requests
import logging
import pandas as pd

# Use project root as base
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
FILE_PATH = os.path.join(RAW_PATH, "funko.csv")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
CLEAN_FILE_PATH = os.path.join(PROCESSED_PATH, "funko_clean.csv")
FRANCHISE_FILE_PATH = os.path.join(PROJECT_ROOT, "utils", "franchise_map.txt")

#extract franchise by mapping(ongoing)
_FRANCHISE_MAP = None

def load_franchise_map():
    global _FRANCHISE_MAP
    if _FRANCHISE_MAP is None:
        try:
            with open(FRANCHISE_FILE_PATH, "r") as f:
                _FRANCHISE_MAP = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Warning: Could not load franchise map: {e}")
            _FRANCHISE_MAP = {}
    return _FRANCHISE_MAP


def extract_franchise(name):
    try:
        with open(FRANCHISE_FILE_PATH, "r") as f:
            franchise_map = json.load(f)

        name_lower = name.lower()
        for franchise, data in franchise_map.items():
            characters = data.get("characters", [])
            if any(keyword.lower() in name_lower for keyword in characters):
                return franchise
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return name.split()[0] if name.split() else "Unknown"

def validate_image_exists(image):
    try:
        response = requests.head(image)
        if response.status_code == 200:
            content_type = response.headers["content-type"]
            if content_type.startswith("image/"):
                return True
            else:
                return False
        else:
            return False
    except requests.exceptions.RequestException:
        return False

def cleaner(df):
    if df.empty:
        return df

    # Remove duplicates
    df = df.drop_duplicates(subset=['name'])

    # Clean price data and normalize text
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df[df['price'] > 0]

    df = df.dropna(subset=['price'])
    df['name'] = df['name'].str.strip()

    # Identifying franchise
    df['franchise'] = df['name'].apply(extract_franchise)

    # Image URL  - currently skipping because of cost
    # df['image'] = df['image'].apply(validate_image_exists)
    # Validate required fields
    df = df.dropna(subset=['name', 'image', 'category'])

    return df

if __name__ == "__main__":
    df = pd.read_csv(FILE_PATH)
    clean_data = cleaner(df)

    clean_data.to_csv(CLEAN_FILE_PATH, index=False)
