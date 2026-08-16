# SVD modelini test setinde degerlendirme (RMSE / MAE)

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

predictions = np.load(SCRIPT_DIR / "predictions.npy")
model = np.load(SCRIPT_DIR / "svd_model.npz")
user_means = model["user_means"]

with open(SCRIPT_DIR / "cf_mappers.pkl", "rb") as f:
    mappers = pickle.load(f)
user_mapper = mappers["user_mapper"]
movie_mapper = mappers["movie_mapper"]

test = pd.read_csv(SCRIPT_DIR / "ratings_test.csv")
train = pd.read_csv(SCRIPT_DIR / "ratings_train.csv")
train_movies = set(train["movieId"])

test = test.copy()
test["user_idx"] = test["userId"].map(user_mapper)
test["movie_idx"] = test["movieId"].map(movie_mapper)
test["seen_in_train"] = test["movieId"].isin(train_movies)

test["pred_svd"] = predictions[test["user_idx"], test["movie_idx"]]
# baseline: her zaman o kullanicinin train ortalamasini tahmin olarak kullan
test["pred_baseline"] = user_means[test["user_idx"]]


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


print("=== TUM test seti (train'de hic gorulmemis filmler dahil) ===")
print(f"SVD      -> RMSE: {rmse(test['rating'], test['pred_svd']):.4f}   MAE: {mae(test['rating'], test['pred_svd']):.4f}")
print(f"Baseline -> RMSE: {rmse(test['rating'], test['pred_baseline']):.4f}   MAE: {mae(test['rating'], test['pred_baseline']):.4f}")

seen = test[test["seen_in_train"]]
print(f"\n=== SADECE train'de gorulen filmler ({len(seen)}/{len(test)} satir) ===")
print(f"SVD      -> RMSE: {rmse(seen['rating'], seen['pred_svd']):.4f}   MAE: {mae(seen['rating'], seen['pred_svd']):.4f}")
print(f"Baseline -> RMSE: {rmse(seen['rating'], seen['pred_baseline']):.4f}   MAE: {mae(seen['rating'], seen['pred_baseline']):.4f}")

unseen = test[~test["seen_in_train"]]
print(f"\n=== SADECE train'de hic gorulmeyen filmler ({len(unseen)}/{len(test)} satir) ===")
print(f"SVD      -> RMSE: {rmse(unseen['rating'], unseen['pred_svd']):.4f}   MAE: {mae(unseen['rating'], unseen['pred_svd']):.4f}")
print(f"Baseline -> RMSE: {rmse(unseen['rating'], unseen['pred_baseline']):.4f}   MAE: {mae(unseen['rating'], unseen['pred_baseline']):.4f}")
