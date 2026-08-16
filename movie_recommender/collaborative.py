"""Mean-centered SVD ile collaborative filtering (scripts/13-18'in paketlenmis hali)."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from . import config


def build_mappers(ratings):
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()
    user_mapper = {uid: i for i, uid in enumerate(user_ids)}
    movie_mapper = {mid: i for i, mid in enumerate(movie_ids)}
    movie_inv_mapper = {i: mid for mid, i in movie_mapper.items()}
    return user_mapper, movie_mapper, movie_inv_mapper


def split_ratings(ratings):
    return train_test_split(ratings, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)


def _build_matrix(ratings, user_mapper, movie_mapper):
    n_users, n_movies = len(user_mapper), len(movie_mapper)
    user_idx = ratings["userId"].map(user_mapper)
    movie_idx = ratings["movieId"].map(movie_mapper)
    return csr_matrix((ratings["rating"], (user_idx, movie_idx)), shape=(n_users, n_movies))


def train_svd(train_ratings, user_mapper, movie_mapper, k=config.SVD_K):
    """Kullanici ortalamasini cikarip (mean-centering) SVD egitir, 0.5-5.0'a clip edilmis
    tam tahmin matrisini dondurur."""
    matrix = _build_matrix(train_ratings, user_mapper, movie_mapper)

    sums = np.array(matrix.sum(axis=1)).flatten()
    counts = np.diff(matrix.indptr)
    user_means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)

    centered_data = matrix.data - user_means[matrix.tocoo().row]
    centered = csr_matrix((centered_data, matrix.indices, matrix.indptr), shape=matrix.shape)

    U, sigma, Vt = svds(centered.astype(float), k=k)
    predictions = np.clip(U @ np.diag(sigma) @ Vt + user_means[:, np.newaxis], 0.5, 5.0)
    return predictions, user_means


def recommend_for_user(user_id, predictions, user_mapper, movie_inv_mapper, movies, all_ratings, k=10):
    if user_id not in user_mapper:
        return []

    user_idx = user_mapper[user_id]
    user_predictions = predictions[user_idx]
    already_watched = set(all_ratings[all_ratings["userId"] == user_id]["movieId"])

    results = []
    for movie_idx in user_predictions.argsort()[::-1]:
        movie_id = movie_inv_mapper[movie_idx]
        if movie_id in already_watched:
            continue
        title = movies.loc[movies["movieId"] == movie_id, "title"].values[0]
        results.append((title, float(user_predictions[movie_idx])))
        if len(results) == k:
            break
    return results
