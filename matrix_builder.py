"""
Build the user-item sparse matrix used for memory-based CF.
"""
import pandas as pd
from scipy.sparse import csr_matrix


def build_user_item_matrix(ratings: pd.DataFrame):
    """
    Returns:
        matrix: scipy sparse matrix (users x items)
        user_map: dict user_id -> row index
        item_map: dict item_id -> col index
        reverse_item_map: dict col index -> item_id
    """
    user_ids = ratings["user_id"].unique()
    item_ids = ratings["item_id"].unique()

    user_map = {uid: i for i, uid in enumerate(user_ids)}
    item_map = {iid: i for i, iid in enumerate(item_ids)}
    reverse_item_map = {i: iid for iid, i in item_map.items()}

    rows = ratings["user_id"].map(user_map)
    cols = ratings["item_id"].map(item_map)
    values = ratings["rating"].values

    matrix = csr_matrix((values, (rows, cols)),
                         shape=(len(user_map), len(item_map)))

    return matrix, user_map, item_map, reverse_item_map


if __name__ == "__main__":
    ratings = pd.read_csv("../data/ratings_clean.csv")
    matrix, user_map, item_map, reverse_item_map = build_user_item_matrix(ratings)
    sparsity = 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    print(f"Matrix shape: {matrix.shape}, sparsity: {sparsity:.4f}")
