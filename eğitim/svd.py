"""
SVD ile Collaborative Filtering - MovieLens (ml-latest-small)

item-based CF'den (collaborative_filtering.py) farkı: burada gercekten
"egitilen" bir model var. Kullanici x film matrisi, matris faktorizasyonu
(SVD) ile kullanici ve film icin kucuk boyutlu gizli ozellik (latent factor)
vektorlerine ayristirilir; tahminler bu vektorlerden yeniden kurulur.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_DIR = "../dataset/ml-latest-small"
CACHE_FILE = Path(__file__).resolve().parent / "cache" / "svd_model.pkl"
TOP_N = 10
MIN_RATINGS = 20  # az oylanan filmlerin latent vektoru az veriyle fit edildigi icin bazen asiri skor alabilir
SVD_K = 10  # gizli boyut sayisi (hiperparametre, egitilmez -> RMSE'ye gore tune edilir)
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data():
    ratings = pd.read_csv(f"{DATASET_DIR}/ratings.csv")
    movies = pd.read_csv(f"{DATASET_DIR}/movies.csv")
    return ratings, movies


def build_mappers(ratings: pd.DataFrame):
    # tum kullanici/filmleri kapsar (train'de olmayanlar matriste sifir satir/sutun olarak durur)
    user_mapper = {uid: i for i, uid in enumerate(ratings["userId"].unique())}
    movie_mapper = {mid: i for i, mid in enumerate(ratings["movieId"].unique())}
    movie_inv_mapper = {i: mid for mid, i in movie_mapper.items()}
    return user_mapper, movie_mapper, movie_inv_mapper


def build_matrix(ratings: pd.DataFrame, user_mapper: dict, movie_mapper: dict):
    n_users, n_movies = len(user_mapper), len(movie_mapper)
    user_idx = ratings["userId"].map(user_mapper)
    movie_idx = ratings["movieId"].map(movie_mapper)
    return csr_matrix((ratings["rating"], (user_idx, movie_idx)), shape=(n_users, n_movies))


def train_svd(train_ratings: pd.DataFrame, user_mapper: dict, movie_mapper: dict, k: int = SVD_K):
    matrix = build_matrix(train_ratings, user_mapper, movie_mapper)

    # mean-centering: 0'lar "kotu puan" degil "bilinmiyor" anlamina gelsin diye
    # her kullanicinin kendi ortalamasi cikarilir, tahminde geri eklenir
    sums = np.array(matrix.sum(axis=1)).flatten()
    counts = np.diff(matrix.indptr)
    user_means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)

    centered_data = matrix.data - user_means[matrix.tocoo().row]
    centered = csr_matrix((centered_data, matrix.indices, matrix.indptr), shape=matrix.shape)

    # matris ~ U . sigma . Vt : U=kullanici, Vt=film gizli vektorleri, sigma=boyut agirliklari
    U, sigma, Vt = svds(centered.astype(float), k=k)
    predictions = np.clip(U @ np.diag(sigma) @ Vt + user_means[:, np.newaxis], 0.5, 5.0)
    return predictions


def train_or_load(train_ratings: pd.DataFrame, user_mapper: dict, movie_mapper: dict,
                   k: int = SVD_K, use_cache: bool = True):
    # egitim maliyetli degil ama tekrar tekrar calistirirken zaman kazandirir
    if use_cache and CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    predictions = train_svd(train_ratings, user_mapper, movie_mapper, k=k)

    if use_cache:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(predictions, f)

    return predictions


def recommend_for_user(user_id: int, predictions, user_mapper: dict, movie_inv_mapper: dict,
                        movies: pd.DataFrame, all_ratings: pd.DataFrame, k: int = TOP_N, min_ratings: int = MIN_RATINGS):
    if user_id not in user_mapper:
        return []

    user_idx = user_mapper[user_id]
    user_predictions = predictions[user_idx]
    already_watched = set(all_ratings[all_ratings["userId"] == user_id]["movieId"])
    well_known = set(all_ratings.groupby("movieId").size().loc[lambda c: c >= min_ratings].index)

    results = []
    for movie_idx in user_predictions.argsort()[::-1]:
        movie_id = movie_inv_mapper[movie_idx]
        if movie_id in already_watched or movie_id not in well_known:
            continue
        title = movies.loc[movies["movieId"] == movie_id, "title"].values[0]
        results.append((title, float(user_predictions[movie_idx])))
        if len(results) == k:
            break
    return results


def evaluate(test_ratings: pd.DataFrame, train_movie_ids: set, predictions, user_mapper: dict, movie_mapper: dict):
    # test'teki film train'de hic gorulmediyse (matriste tum sutunu 0) tahmin anlamsizdir, elenir
    test = test_ratings[test_ratings["movieId"].isin(train_movie_ids)].copy()
    test["user_idx"] = test["userId"].map(user_mapper)
    test["movie_idx"] = test["movieId"].map(movie_mapper)

    y_true = test["rating"].values
    y_pred = predictions[test["user_idx"], test["movie_idx"]]

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return float(rmse), float(mae), len(test)


def main():
    ratings, movies = load_data()
    user_mapper, movie_mapper, movie_inv_mapper = build_mappers(ratings)

    train_ratings, test_ratings = train_test_split(ratings, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_movie_ids = set(train_ratings["movieId"])

    predictions = train_or_load(train_ratings, user_mapper, movie_mapper, k=SVD_K)

    rmse, mae, n = evaluate(test_ratings, train_movie_ids, predictions, user_mapper, movie_mapper)
    print(f"===== SVD (k={SVD_K}) Değerlendirme =====")
    print(f"RMSE: {rmse:.4f}   MAE: {mae:.4f}   (n={n} tahmin)\n")

    for user_id in [1, 414]:
        print(f"===== Kullanıcı {user_id} için SVD Önerisi =====")
        results = recommend_for_user(user_id, predictions, user_mapper, movie_inv_mapper, movies, ratings)
        df = pd.DataFrame(results, columns=["title", "predicted_rating"])
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
