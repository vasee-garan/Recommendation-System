import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix


# -------------------------------------------------------------------
# Load toy data (browsing + purchasing history)
# -------------------------------------------------------------------
def load_data():
    products = pd.DataFrame({
        "productId": [1, 2, 3, 4, 5, 6, 7],
        "name": [
            "Wireless Mouse", "Gaming Laptop", "Office Chair", "Bluetooth Speaker",
            "Mechanical Keyboard", "Running Shoes", "Smartphone"
        ],
        "category": [
            "Electronics Accessories", "Electronics Computers", "Furniture",
            "Electronics Audio", "Electronics Accessories", "Sportswear", "Electronics Mobile"
        ],
        "description": [
            "Ergonomic wireless mouse with USB receiver",
            "High performance gaming laptop with powerful GPU",
            "Comfortable office chair with lumbar support",
            "Portable Bluetooth speaker with deep bass",
            "RGB backlit mechanical keyboard for gaming",
            "Lightweight running shoes for men and women",
            "Latest smartphone with high-resolution camera"
        ]
    })

    users = pd.DataFrame({
        "userId": [1, 2, 3, 4]
    })

    # Browsing + purchasing history
    interactions = pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 3, 3, 4, 4],
        "productId": [1, 2, 4, 2, 3, 1, 5, 6, 7],
        "interaction": ["purchase", "browse", "purchase",
                        "purchase", "browse",
                        "browse", "purchase",
                        "browse", "purchase"]
    })

    return products, users, interactions


# -------------------------------------------------------------------
# Popularity-based recommender
# -------------------------------------------------------------------
class PopularityRecommender:
    def __init__(self):
        self.popularity = None

    def fit(self, interactions: pd.DataFrame):
        self.popularity = (
            interactions.groupby("productId")["interaction"]
            .count()
            .sort_values(ascending=False)
        )
        return self

    def recommend(self, user_id: int, k: int = 5) -> List[int]:
        return list(self.popularity.head(k).index)


# -------------------------------------------------------------------
# Content-based recommender (TF-IDF on description + category)
# -------------------------------------------------------------------
class ContentRecommender:
    def __init__(self):
        self.sim_matrix = None
        self.products = None

    def fit(self, products: pd.DataFrame):
        vectorizer = TfidfVectorizer(stop_words="english")
        features = vectorizer.fit_transform(
            products["name"] + " " + products["category"] + " " + products["description"]
        )
        self.sim_matrix = cosine_similarity(features)
        self.products = products.set_index("productId")
        return self

    def recommend(self, user_id: int, interactions: pd.DataFrame, k: int = 5) -> List[int]:
        seen = interactions.loc[interactions.userId == user_id, "productId"].tolist()
        if not seen:
            return []

        sim_scores = np.zeros(len(self.products))
        for pid in seen:
            idx = self.products.index.get_loc(pid)
            sim_scores += np.asarray(self.sim_matrix[idx])  # ensure np.array

        sim_scores = sim_scores / len(seen)
        ranked = np.argsort(sim_scores)[::-1]
        recs = [self.products.index[i] for i in ranked if self.products.index[i] not in seen]
        return recs[:k]


# -------------------------------------------------------------------
# Collaborative filtering with matrix factorization
# -------------------------------------------------------------------
class MFRecommender:
    def __init__(self, n_components: int = 32, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.inv_user_map = None
        self.inv_item_map = None

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame):
        users = interactions["userId"].unique()
        items = products["productId"].unique()
        user_map = {u: i for i, u in enumerate(users)}
        item_map = {p: i for i, p in enumerate(items)}
        inv_user_map = {i: u for u, i in user_map.items()}
        inv_item_map = {i: p for p, i in item_map.items()}

        # implicit feedback matrix
        rows = [user_map[u] for u in interactions.userId]
        cols = [item_map[p] for p in interactions.productId]
        vals = [2 if t == "purchase" else 1 for t in interactions.interaction]
        X = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))

        # Fix n_components based on product count
        n_items = X.shape[1]
        n_comp = min(self.n_components, max(1, n_items - 1))

        svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        X_log = X.copy()
        X_log.data = np.log1p(X_log.data)

        U = svd.fit_transform(X_log)      # users × k
        V = svd.components_.T             # items × k

        # Convert to ndarray explicitly
        self.user_factors = np.asarray(U) / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)
        self.item_factors = np.asarray(V) / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        self.inv_user_map = inv_user_map
        self.inv_item_map = inv_item_map
        return self

    def recommend(self, user_id: int, interactions: pd.DataFrame, k: int = 5) -> List[int]:
        if user_id not in self.inv_user_map.values():
            return []
        u_idx = {u: i for i, u in self.inv_user_map.items()}[user_id]
        scores = self.item_factors @ self.user_factors[u_idx]

        seen = interactions.loc[interactions.userId == user_id, "productId"].tolist()
        ranked = np.argsort(scores)[::-1]
        recs = [self.inv_item_map[i] for i in ranked if self.inv_item_map[i] not in seen]
        return recs[:k]


# -------------------------------------------------------------------
# Hybrid recommender (weighted blend)
# -------------------------------------------------------------------
class HybridRecommender:
    def __init__(self, products, interactions, users, w_pop=0.2, w_content=0.3, w_cf=0.5):
        self.products = products
        self.interactions = interactions
        self.users = users
        self.pop = PopularityRecommender().fit(interactions)
        self.content = ContentRecommender().fit(products)
        self.cf = MFRecommender().fit(interactions, products)
        self.w_pop = w_pop
        self.w_content = w_content
        self.w_cf = w_cf

    def recommend(self, user_id: int, k: int = 5) -> List[str]:
        pop_rec = self.pop.recommend(user_id, k=20)
        cont_rec = self.content.recommend(user_id, self.interactions, k=20)
        cf_rec = self.cf.recommend(user_id, self.interactions, k=20)

        scores = {}
        for i, recs in enumerate([pop_rec, cont_rec, cf_rec]):
            weight = [self.w_pop, self.w_content, self.w_cf][i]
            for rank, pid in enumerate(recs):
                scores[pid] = scores.get(pid, 0) + weight / (rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        product_ids = [pid for pid, _ in ranked[:k]]
        return self.products.set_index("productId").loc[product_ids, "name"].tolist()


# -------------------------------------------------------------------
# Demo run
# -------------------------------------------------------------------
if __name__ == "__main__":
    products, users, interactions = load_data()
    recommender = HybridRecommender(products, interactions, users)

    for uid in users["userId"]:
        print(f"\nTop recommendations for user {uid}:")
        print(recommender.recommend(uid, k=3))
