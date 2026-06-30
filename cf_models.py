"""
Two collaborative filtering models:
1. Item-item KNN (cosine similarity) - memory-based
2. SVD matrix factorization - model-based (via scikit-surprise)
"""
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split


# ---------- Item-Item KNN ----------

class ItemItemKNN:
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")

    def fit(self, item_user_matrix):
        """item_user_matrix: sparse matrix shaped (items x users)"""
        self.matrix = item_user_matrix
        self.model.fit(item_user_matrix)

    def similar_items(self, item_idx, k=None):
        k = k or self.n_neighbors
        distances, indices = self.model.kneighbors(
            self.matrix[item_idx], n_neighbors=k + 1
        )
        # skip the first result (item itself)
        return list(zip(indices.flatten()[1:], 1 - distances.flatten()[1:]))


# ---------- SVD (Surprise) ----------

def train_svd(ratings: pd.DataFrame, n_factors=50, test_size=0.2, random_state=42):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings[["user_id", "item_id", "rating"]], reader
    )
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

    model = SVD(n_factors=n_factors, random_state=random_state)
    model.fit(trainset)

    return model, testset


if __name__ == "__main__":
    ratings = pd.read_csv("../data/ratings_clean.csv")
    model, testset = train_svd(ratings)
    print("SVD model trained.")
