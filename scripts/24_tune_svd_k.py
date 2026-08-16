# SVD icin en iyi k (gizli faktor sayisi) degerini RMSE ile secme

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

SCRIPT_DIR = Path(__file__).resolve().parent

with open(SCRIPT_DIR / "cf_mappers.pkl", "rb") as f:
    mappers = pickle.load(f)
user_mapper = mappers["user_mapper"]
movie_mapper = mappers["movie_mapper"]

train = pd.read_csv(SCRIPT_DIR / "ratings_train.csv")
test = pd.read_csv(SCRIPT_DIR / "ratings_test.csv")

n_users = len(user_mapper)
n_movies = len(movie_mapper)

user_index = train["userId"].map(user_mapper)
movie_index = train["movieId"].map(movie_mapper)
train_matrix = csr_matrix((train["rating"], (user_index, movie_index)), shape=(n_users, n_movies))

sums = np.array(train_matrix.sum(axis=1)).flatten()
counts = np.diff(train_matrix.indptr)
user_means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)

centered_data = train_matrix.data - user_means[train_matrix.tocoo().row]
centered_matrix = csr_matrix((centered_data, train_matrix.indices, train_matrix.indptr), shape=train_matrix.shape)

# degerlendirme icin: sadece train'de gorulen filmler (adim 5'teki ayni gerekce)
train_movies = set(train["movieId"])
test_seen = test[test["movieId"].isin(train_movies)].copy()
test_seen["user_idx"] = test_seen["userId"].map(user_mapper)
test_seen["movie_idx"] = test_seen["movieId"].map(movie_mapper)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


results = []
for k in [5, 10, 20, 30, 50, 100]:
    U, sigma, Vt = svds(centered_matrix.astype(float), k=k)
    preds = U @ np.diag(sigma) @ Vt + user_means[:, np.newaxis]
    preds = np.clip(preds, 0.5, 5.0)

    pred_values = preds[test_seen["user_idx"], test_seen["movie_idx"]]
    score = rmse(test_seen["rating"], pred_values)
    results.append((k, score))
    print(f"k={k:4d}  ->  RMSE: {score:.4f}")

best_k, best_rmse = min(results, key=lambda x: x[1])
print(f"\nEn iyi k: {best_k} (RMSE: {best_rmse:.4f})")

# en iyi k ile modeli yeniden egitip mevcut dosyalarin uzerine yazalim
# (boylece hybrid asamasindaki predictions.npy de guncellenmis olur)
U, sigma, Vt = svds(centered_matrix.astype(float), k=best_k)
predictions = np.clip(U @ np.diag(sigma) @ Vt + user_means[:, np.newaxis], 0.5, 5.0)

np.savez(SCRIPT_DIR / "svd_model.npz", U=U, sigma=sigma, Vt=Vt, user_means=user_means)
np.save(SCRIPT_DIR / "predictions.npy", predictions)
print(f"\nEn iyi model (k={best_k}) ile svd_model.npz ve predictions.npy guncellendi.")
