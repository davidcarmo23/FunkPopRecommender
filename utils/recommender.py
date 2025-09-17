"""
Content-based recommender for FunkoPops.
- Load cleaned data
- Build TF-IDF model
- Recommend similar Pops
"""
import os
import json
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from joblib import dump, load

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
FRANCHISE_FILE_PATH = os.path.join(PROJECT_ROOT, "utils", "franchise_map.txt")


class FunkoPopRecommender:
    def __init__(self, df):
        self.df = df
        self.df["features"] = (
                df["name"] + " " +
                df["franchise"] + " " +
                df["category"] + " " +
                df["price"].apply(lambda x: "expensive" if x > 50 else "affordable")
        )
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["features"])
        # Remove the full cosine similarity matrix - compute on demand
        self.indices = pd.Series(df.index, index=df['name'].to_dict())

        # cache frequently accessed items
        self._similarity_cache = {}
        self._max_cache_size = 100

    def save_model(self, filepath):
        dump(self, filepath)

    @classmethod
    def load_model(cls, filepath):
        return load(filepath)

    def _calculate_popularity_score(self, items_df):
        """Calculate popularity score based on multiple factors"""
        scores = pd.Series(0.0, index=items_df.index)

        # Price tier scoring
        price_scores = items_df['price'].apply(self._price_popularity_score)
        scores += price_scores * 0.4

        # Franchise popularity
        franchise_scores = items_df['franchise'].apply(self._franchise_popularity_score)
        scores += franchise_scores * 0.3

        # Item frequency in dataset (more often more popular)
        name_counts = items_df['name'].value_counts()
        frequency_scores = items_df['name'].map(name_counts) / name_counts.max()
        scores += frequency_scores * 0.2

        # Small random factor for discovery and tie-breaking
        np.random.seed(hash(str(items_df.index.tolist())) % 2 ** 32)
        scores += np.random.random(len(items_df)) * 0.1

        return scores

    def _price_popularity_score(self, price):
        """Score based on price range"""
        if price < 15:
            return 0.9  # Budget items
        elif price < 30:
            return 1.0  # Sweet spot
        elif price < 60:
            return 0.7  # Premium
        else:
            return 0.4  # Collector items

    def _franchise_popularity_score(self, franchise):
        try:
            with open(FRANCHISE_FILE_PATH, "r") as f:
                franchise_map = json.load(f)
            return franchise_map.get(franchise, {}).get("popularity", 0.5)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0.5

    def _find_close_matches(self, name, max_matches=3):
        """Find close matches for typos/variations"""
        from difflib import get_close_matches
        all_names = list(self.indices.keys())
        return get_close_matches(name, all_names, n=max_matches, cutoff=0.6)

    def _compute_similarity_for_item(self, idx):
        """Compute similarity score for item on demand"""
        if idx in self._similarity_cache:
            return self._similarity_cache[idx]

        item_vector = self.tfidf_matrix[idx]
        similarities = linear_kernel(item_vector, self.tfidf_matrix).flatten()

        # cache the result
        if len(self._similarity_cache) >= self._max_cache_size:
            # remove oldest
            oldest_key = next(iter(self._similarity_cache))
            del self._similarity_cache[oldest_key]

        self._similarity_cache[idx] = similarities
        return similarities

    def _compute_batch_similarities(self, indices):
        """Compute similarities for item batches"""
        cached_scores = []
        uncached_indices = []

        for idx in indices:
            if idx in self._similarity_cache:
                cached_scores.append(self._similarity_cache[idx])
            else:
                uncached_indices.append(idx)

        # compute just uncached items
        if uncached_indices:
            batch_vectors = self.tfidf_matrix[uncached_indices]
            batch_similarities = linear_kernel(batch_vectors, self.tfidf_matrix)

            # cache individual results
            for i, idx in enumerate(uncached_indices):
                similarities = batch_similarities[i]
                if len(self._similarity_cache) >= self._max_cache_size:
                    oldest_key = next(iter(self._similarity_cache))
                    del self._similarity_cache[oldest_key]
                self._similarity_cache[idx] = similarities
                cached_scores.append(similarities)

        # aggregate all scores
        if len(cached_scores) == 1:
            return cached_scores[0]
        else:
            aggregated = np.zeros(len(self.df))
            for scores in cached_scores:
                aggregated += scores
            return aggregated / len(cached_scores)  # normalized

    def clear_cache(self):
        """Clear similarity cache to free memory"""
        self._similarity_cache.clear()

    def recommend_by_name(self, pop_name, top_n=5, similarity_threshold=0.1):
        """applied lazy computation"""
        if pop_name not in self.indices:
            # Try fuzzy matching for typos/variations
            close_matches = self._find_close_matches(pop_name)
            if close_matches:
                return pd.DataFrame(), f"Item '{pop_name}' not found. Did you mean: {', '.join(close_matches[:3])}?"
            else:
                return pd.DataFrame(), f"Item '{pop_name}' not found in database."

        idx = self.indices[pop_name]

        # compute similarities
        similarities = self._compute_similarity_for_item(idx)

        sim_scores = [(i, score) for i, score in enumerate(similarities) if i != idx]
        sim_scores.sort(key=lambda x: x[1], reverse=True)

        # filter by similarity
        good_matches = [s for s in sim_scores[:top_n + 10] if s[1] >= similarity_threshold]

        if len(good_matches) >= top_n:
            # enough matches
            final_scores = good_matches[:top_n]
            return self.df.ilock[[i[0] for i in final_scores]], None
        elif len(good_matches) > 0:
            # some matches
            good_recs = self.df.iloc[[i[0] for i in good_matches]]
            needed = top_n - len(good_matches)

            original_item = self.df.iloc[idx]
            category_recs, _ = self.recommend_by_category(
                original_item['franchise'],
                top_n=needed,
                exclude_owned=[pop_name] + good_recs['name'].toList(),
            )
            combined = pd.concat([good_recs, category_recs], ignore_index=True)
            return combined, f"Limited similar items found. Added popular items from the same franchise"
        else:
            # no matches
            original_item = self.df.iloc[idx]
            category_recs, msg = self.recommend_by_category(
                original_item['franchise'],
                top_n=top_n,
                exclude_owned=[pop_name]
            )
            return category_recs, f"No similar items found. Showing popular items from same franchise ({original_item['franchise']})."

    def recommend_by_owned(self, owned_list, top_n=5, similarity_threshold=0.05):
        """applied batch processing to improve performance"""
        owned_indices = [self.indices[name] for name in owned_list if name in self.indices]
        if not owned_indices:
            # None of the owned items were found in db
            return pd.DataFrame()

        # Accumulate scores correctly
        scores = self._compute_batch_similarities(owned_indices)
        scores[owned_indices] = 0

        # filter with threshold
        top_indices = scores.argsort()[::-1][:top_n * 2]
        top_scores = scores[top_indices]
        good_indices = top_indices[top_scores >= similarity_threshold]

        if len(good_indices) >= top_n:
            #sufficient recommendations
            final_indices = good_indices[:top_n]
            return self.df.iloc[final_indices], None

        elif len(good_indices) > 0:
            #some recommendations fill based on franchise popularity
            good_recs = self.df.iloc[good_indices]
            needed = top_n - len(good_indices)

            owned_franchises = self.df.iloc[owned_indices]['franchise'].unique()
            category_recs = pd.DataFrame()
            exclude_names = owned_list + good_recs['name'].tolist()

            for franchise in owned_franchises[:2]:
                if needed <= 0:
                    break
                franchise_recs, _ = self.recommend_by_category(
                    franchise,
                    top_n=needed,
                    exclude_owned=exclude_names
                )
                if not franchise_recs.empty:
                    category_recs = pd.concat([category_recs, franchise_recs], ignore_index=True)
                    exclude_names.extend(franchise_recs['name'].tolist())
                    needed = top_n - len(good_recs) - len(category_recs)

            combined = pd.concat([good_recs, category_recs.head(top_n - len(good_recs))], ignore_index=True)
            return combined, "Limited similar items found. Added popular items from your favorite franchises."

        else:
            # no recommendations found fill with popular franchise items
            owned_franchises = self.df.iloc[owned_indices]['franchise'].value_counts()
            most_common_franchise = owned_franchises.index[0]

            category_recs, msg = self.recommend_by_category(
                most_common_franchise,
                top_n=top_n,
                exclude_owned=owned_list
            )
            return category_recs, f"No similar patterns found. Showing popular items from your most collected franchise ({most_common_franchise})."

    def recommend_by_category(self, category, top_n=5, exclude_owned=None, min_items=3):
        """
            check popularity based on users pop with the same category
            and recommend it
        """

        if exclude_owned is None:
            exclude_owned = []

        # Filter
        category_lower = category.lower()
        category_items = self.df[
            (self.df['category'].str.lower().str.contains(category_lower, na=False)) |
            (self.df['franchise'].str.lower().str.contains(category_lower, na=False)) |
            (self.df['name'].str.lower().str.contains(category_lower, na=False))
            ].copy()

        if len(category_items) < min_items:
            return pd.DataFrame(), f"Not enough items in category '{category}' (found {len(category_items)}, need {min_items})"

        if exclude_owned:
            category_items = category_items[~category_items['name'].isin(exclude_owned)]

        if category_items.empty:
            return pd.DataFrame(), f"No new items to recommend in category '{category}' (all items already owned)"

        # Popularity scoring
        category_items['popularity_score'] = self._calculate_popularity_score(category_items)

        # Sort by popularity and return top N
        recommendations = category_items.nlargest(min(top_n, len(category_items)), 'popularity_score')

        # Clean up and return necessary columns
        result_cols = ['name', 'price', 'image', 'category', 'franchise']
        if 'popularity_score' in recommendations.columns:
            recommendations = recommendations.drop('popularity_score', axis=1)

        return recommendations[result_cols].reset_index(drop=True), None


if __name__ == "__main__":
    f = FunkoPopRecommender()
