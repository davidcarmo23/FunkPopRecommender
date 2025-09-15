import streamlit as st
import pandas as pd
from utils.recommender import FunkoPopRecommender
import os

# Paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
FUNKO_FILE = os.path.join(PROCESSED_PATH, "funko_clean.csv")
PRICE_LOG_FILE = os.path.join(RAW_PATH, "price_changes.csv")

# Load datasets
df = pd.read_csv(FUNKO_FILE)
price_log = pd.read_csv(PRICE_LOG_FILE) if os.path.exists(PRICE_LOG_FILE) else pd.DataFrame()

# Initialize recommender
recommender = FunkoPopRecommender(df)

st.title("🎉 FunkoPop Recommender")

st.markdown("## Single Pop Recommendation")
single_pop = st.selectbox("Select a Pop you own or like:", df["name"].tolist())

if single_pop:
    recs = recommender.recommend_by_name(single_pop, top_n=5)
    st.write("Similar Pops you might like:")
    for _, row in recs.iterrows():
        st.image(row["image"], width=100)
        st.write(f"**{row['name']}** — {row['category']} — €{row['price']}")

st.markdown("---")
st.markdown("## Personalized Recommendation (Owned Pops)")
owned_pops = st.multiselect(
    "Select all the Pops you own:",
    df["name"].tolist()
)

if owned_pops:
    recs = recommender.recommend_by_owned(owned_pops, top_n=5)
    st.write("Recommended Pops based on your collection:")
    for _, row in recs.iterrows():
        # Check for discounts
        discount = ""
        if not price_log.empty:
            changes = price_log[price_log["name"] == row["name"]]
            if not changes.empty:
                latest_change = changes.iloc[-1]
                if latest_change["new_price"] < latest_change["old_price"]:
                    discount = f"🔥 Discounted from €{latest_change['old_price']}!"

        st.image(row["image"], width=100)
        st.write(f"**{row['name']}** — {row['category']} — €{row['price']} {discount}")
