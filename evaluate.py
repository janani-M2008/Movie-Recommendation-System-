"""
Evaluate CF models: RMSE/MAE (rating prediction) and Precision@K/Recall@K
(top-N recommendation quality).
"""
from collections import defaultdict
import pandas as pd
from surprise import accuracy

from cf_models import train_svd


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Computes Precision@K and Recall@K for each user, then averages.
    `predictions` is the list returned by model.test(testset).
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in ratings[:k])
        n_rel_and_rec_k = sum(
            (true_r >= threshold) and (est >= threshold)
            for (est, true_r) in ratings[:k]
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    avg_precision = sum(precisions.values()) / len(precisions)
    avg_recall = sum(recalls.values()) / len(recalls)
    return avg_precision, avg_recall


def popularity_baseline_rmse(ratings: pd.DataFrame):
    """Predict the global mean rating for every entry — simplest baseline."""
    global_mean = ratings["rating"].mean()
    mse = ((ratings["rating"] - global_mean) ** 2).mean()
    return mse ** 0.5


if __name__ == "__main__":
    ratings = pd.read_csv("../data/ratings_clean.csv")

    baseline_rmse = popularity_baseline_rmse(ratings)
    print(f"Popularity baseline RMSE: {baseline_rmse:.4f}")

    model, testset = train_svd(ratings)
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    precision, recall = precision_recall_at_k(predictions, k=10)

    print(f"SVD RMSE: {rmse:.4f}")
    print(f"SVD MAE: {mae:.4f}")
    print(f"SVD Precision@10: {precision:.4f}")
    print(f"SVD Recall@10: {recall:.4f}")
