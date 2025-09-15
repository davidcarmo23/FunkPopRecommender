"""
Content-based recommender for FunkoPops.
- Load cleaned data
- Build TF-IDF model
- Recommend similar Pops
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np

class FunkoPopRecommender:
    def __init__(self, df):
        self.df = df
        self.df["features"] = df["name"] + " " + df["franchise"]
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["features"])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(df.index, index=df["name"]).to_dict()

    def recommend_by_name(self, pop_name, top_n=5):
        if pop_name not in self.indices:
            return  pd.DataFrame()

        idx = self.indices[pop_name]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n]
        return self.df.iloc[[i[0] for i in sim_scores]]

    def recommend_by_owned(self, owned_list, top_n=5):
        owned_indices = [self.indices[name] for name in owned_list if name in self.indices]
        scores = np.zeros(len(self.df))
        for idx in owned_indices:
            scores = self.cosine_sim[idx]
        scores[owned_indices] = 0
        top_indices = scores.argsort()[::-1][:top_n]
        return self.df.iloc[top_indices]

    def recommend_by_category(self, pop_category, top_n=5):
        if pop_category not in self.indices:
            return pd.DataFrame()
        """
            check popularity based on users pop with the same category 
            and recommend it
        """


if __name__ == "__main__":
    f = FunkoPopRecommender()
    print(f.recommend(10))