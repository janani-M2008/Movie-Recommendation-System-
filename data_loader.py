"""
Download and load the MovieLens 100k dataset.
"""
import os
import zipfile
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_movielens():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")

    if not os.path.exists(zip_path):
        print("Downloading MovieLens 100k...")
        r = requests.get(ML_URL, timeout=30)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)

    extract_dir = os.path.join(DATA_DIR, "ml-100k")
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)

    print(f"Data ready at {extract_dir}")
    return extract_dir


def load_ratings(extract_dir):
    cols = ["user_id", "item_id", "rating", "timestamp"]
    return pd.read_csv(
        os.path.join(extract_dir, "u.data"), sep="\t", names=cols
    )


def load_movies(extract_dir):
    cols = [
        "item_id", "title", "release_date", "video_release_date", "imdb_url"
    ] + [f"genre_{i}" for i in range(19)]
    return pd.read_csv(
        os.path.join(extract_dir, "u.item"),
        sep="|", names=cols, encoding="latin-1"
    )


def clean_ratings(ratings, min_user_ratings=5, min_item_ratings=5):
    """Drop users/items with too few ratings to reduce sparsity noise."""
    user_counts = ratings["user_id"].value_counts()
    item_counts = ratings["item_id"].value_counts()

    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_items = item_counts[item_counts >= min_item_ratings].index

    return ratings[
        ratings["user_id"].isin(valid_users)
        & ratings["item_id"].isin(valid_items)
    ].reset_index(drop=True)


if __name__ == "__main__":
    extract_dir = download_movielens()
    ratings = load_ratings(extract_dir)
    movies = load_movies(extract_dir)
    ratings = clean_ratings(ratings)

    print(f"Ratings: {len(ratings)} | Users: {ratings['user_id'].nunique()} "
          f"| Movies: {ratings['item_id'].nunique()}")

    ratings.to_csv(os.path.join(DATA_DIR, "ratings_clean.csv"), index=False)
    movies.to_csv(os.path.join(DATA_DIR, "movies_clean.csv"), index=False)
