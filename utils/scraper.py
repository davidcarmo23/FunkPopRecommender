import os
import time
import random
import requests
import pandas as pd
from datetime import datetime

SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FILE_PATH = os.path.join(SAVE_PATH, "funko.csv")
PRICE_LOG_PATH = os.path.join(SAVE_PATH, "price_changes.csv")

def scrape_funko(base_url="https://funkoeurope.com/collections/all/products.json",
                 max_pages=20):
    os.makedirs(SAVE_PATH, exist_ok=True)

    existing_df = pd.read_csv(FILE_PATH) if os.path.exists(FILE_PATH) else pd.DataFrame(columns=["name", "price", "image", "category"])
    price_log = pd.read_csv(PRICE_LOG_PATH) if os.path.exists(PRICE_LOG_PATH) else pd.DataFrame(columns=["name", "old_price", "new_price", "timestamp"])

    all_products = []

    for page in range(1, max_pages + 1):
        url = f"{base_url}?page={page}"
        print(f"Fetching: {url}")
        resp = requests.get(url)


        if resp.status_code != 200:
            print(f"⚠️ Failed to fetch page {page}")
            break

        data = resp.json()
        products = data.get("products", [])

        if not products:  # stop if no more products
            print("No more products found, stopping pagination.")
            break

        for product in products:
            name = product["title"]
            vendor = product["vendor"]
            image = product["images"][0]["src"] if product["images"] else None
            price = float(product["variants"][0]["price"]) if product["variants"] else None

            all_products.append({
                "name": name,
                "price": price,
                "image": image,
                "category": vendor
            })


        time.sleep(random.uniform(5, 10))

    new_df = pd.DataFrame(all_products)

    # Merge with existing
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset=["name"], keep="last", inplace=True)

    # Track price changes
    for _, row in new_df.iterrows():
        name, new_price = row["name"], row["price"]
        old_entry = existing_df[existing_df["name"] == name]
        if not old_entry.empty:
            old_price = float(old_entry.iloc[0]["price"])
            if old_price != new_price:
                print(f"💰 Price change detected for {name}: {old_price} → {new_price}")
                price_log = pd.concat([
                    price_log,
                    pd.DataFrame([{
                        "name": name,
                        "old_price": old_price,
                        "new_price": new_price,
                        "timestamp": datetime.now().isoformat()
                    }])
                ], ignore_index=True)

    # Save updated datasets
    combined_df.to_csv(FILE_PATH, index=False)
    price_log.to_csv(PRICE_LOG_PATH, index=False)

    print(f"✅ Saved {len(combined_df)} products to {FILE_PATH}")
    print(f"📓 Price log updated: {len(price_log)} entries total")

    return combined_df, price_log


if __name__ == "__main__":
    df, price_changes = scrape_funko()
    print(df.head())
