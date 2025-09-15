# FunkoPop Recommender 🎉
Simple Streamlit frontend for a recommendation system for Funko Pops

## Features:
1. Funko name based recommendation
2. List of owned Funko based recommendation

## Future feature:
1. Category based recommendation
2. **CRITICAL** Storing trained objects to not lose recommender progress
2. User features
   3. Simple authentication to allow multiple users:
      4. Allows to store data about Funko Pop ownership rates and popularity which improves recommendation
      5. Top n most popular Funko Pops (by sorting: week/month/year)

## Steps
1. Scrape data (`scraper.py`)
2. Clean & preprocess (`cleaner.py`)
3. Train recommender (`recommender.py`)
4. Demo in Jupyter or Streamlit (`app.py`)

The existing notebooks give a more detailed outlook to the process.
They allow a better understanding to the process and reasoning behind the methods