"""
Generate top-N recommendations for a given user.
Falls back to popularity-based recommendations for cold-start users
(users with no/few ratings).
"""
import pandas as pd


def get_popular_items(ratings: pd.DataFrame, movies: pd.DataFrame, n=10, min_ratings=20):
    """Most popular items by average rating, filtered for a minimum count."""
    stats = ratings.groupby("item_id")["rating"].agg(["mean", "count"])
    stats = stats[stats["count"] >= min_ratings]
    top = stats.sort_values("mean", ascending=False).head(n)
    return movies[movies["item_id"].isin(top.index)][["item_id", "title"]]


def recommend_for_user(user_id, model, ratings, movies, n=10):
    """
    Use the trained SVD model to predict ratings for all items the user
    hasn't rated yet, and return the top-N.
    """
    is_cold_start = user_id not in ratings["user_id"].values
    if is_cold_start:
        return get_popular_items(ratings, movies, n=n)

    all_items = movies["item_id"].unique()
    rated_items = ratings[ratings["user_id"] == user_id]["item_id"].values
    unrated_items = [i for i in all_items if i not in rated_items]

    predictions = [
        (item_id, model.predict(user_id, item_id).est)
        for item_id in unrated_items
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_ids = [item_id for item_id, _ in predictions[:n]]

    return movies[movies["item_id"].isin(top_ids)][["item_id", "title"]]


if __name__ == "__main__":
    from cf_models import train_svd

    ratings = pd.read_csv("../data/ratings_clean.csv")
    movies = pd.read_csv("../data/movies_clean.csv")

    model, _ = train_svd(ratings)
    recs = recommend_for_user(user_id=1, model=model, ratings=ratings, movies=movies)
    print(recs)
