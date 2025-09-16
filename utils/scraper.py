import os
import time
import random
import requests
import logging
import pandas as pd
from datetime import datetime

from requests import JSONDecodeError


SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FILE_PATH = os.path.join(SAVE_PATH, "funko.csv")
PRICE_LOG_PATH = os.path.join(SAVE_PATH, "price_changes.csv")

def scrape_funko(base_url="https://funkoeurope.com/collections/all/products.json",
                 max_pages=20):
    try:
        os.makedirs(SAVE_PATH, exist_ok=True)

        existing_df = pd.read_csv(FILE_PATH) if os.path.exists(FILE_PATH) else pd.DataFrame(columns=["name", "price", "image", "category"])
        price_log = pd.read_csv(PRICE_LOG_PATH) if os.path.exists(PRICE_LOG_PATH) else pd.DataFrame(columns=["name", "old_price", "new_price", "timestamp"])

        all_products = []

        logging.info(f"Starting scrape of {max_pages} pages")
        for page in range(1, max_pages + 1):
            url = f"{base_url}?page={page}"
            print(f"Fetching: {url}")
            resp = requests.get(url)


            if resp.status_code != 200:
                print(f"⚠️ Failed to fetch page {page}")
                logging.warning(f"HTTP {resp.status_code} error on page {page}")
                break
            try:
                data = resp.json()
            except JSONDecodeError:
                print(resp.status_code)
                logging.error("JSON parsing failed", exc_info=True)
                break

            products = data.get("products", [])

            if not products:  # stop if no more
                print("No more products found, stopping pagination.")
                break

            for product in products:
                name = product.get('title')
                vendor = product.get('vendor')
                image = product.get('images', [{}])[0].get("src") if product.get('images') else None

                if product.get("price") is None:
                    price = None
                elif not isinstance(product.get("price"), (int, float)):
                    try:
                        price = float(product.get('price'))
                    except ValueError:
                        price = None
                else:
                    price = product.get("price")

                all_products.append({
                    "name": name,
                    "price": price,
                    "image": image,
                    "category": vendor
                })


            time.sleep(random.uniform(5, 10))

        new_df = pd.DataFrame(all_products)
        # Remove items without pricing
        new_df = new_df.dropna(subset=["price"])

        # Merge with existing
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["name"], keep="last", inplace=True)

        # Track price changes
        price_changes = []
        for _, row in new_df.iterrows():
            name, new_price = (row["name"] if row["name"] else False), (row["price"] if row["price"] else False)
            if name and new_price:
                old_entry = existing_df[existing_df["name"] == name]
                if not old_entry.empty:
                    old_price = float(old_entry.iloc[0]["price"])
                    if old_price != new_price:
                        price_changes.append({
                            "name": name,
                            "old_price": old_price,
                            "new_price": new_price,
                            "timestamp": datetime.now().isoformat()
                        })
        if price_changes:
            logging.info(f"Found {len(price_changes)} price changes")
            new_changes_df = pd.DataFrame(price_changes)
            price_log = pd.concat([price_log, new_changes_df], ignore_index=True)

        # Save updated datasets
        combined_df.to_csv(FILE_PATH, index=False)
        price_log.to_csv(PRICE_LOG_PATH, index=False)

        print(f"✅ Saved {len(combined_df)} products to {FILE_PATH}")
        print(f"📓 Price log updated: {len(price_log)} entries total")

        return combined_df, price_log
    except Exception as e:
        logging.exception(e)
        print("An error occurred while scraping funko.")

if __name__ == "__main__":
    df, price_changes = scrape_funko()
    print(df.head())
