"""
Değerlendirme Metrikleri - MovieLens (ml-latest-small)

eğitim klasöründeki tüm modelleri (content-based, item-based CF, SVD, MF,
NCF, hybrid) aynı train/test split ve aynı kullanıcı örneklemi üzerinde
kıyaslar:
- RMSE / MAE  -> puan tahmini ne kadar isabetli (rating üreten modeller)
- Precision@K / Recall@K -> önerilen top-K filmin ne kadarı kullanıcının
  gerçekten (test setinde) yüksek puan verdiği filmlerle örtüşüyor (hepsi)

svd.py, mf.py, ncf.py'nin kendi evaluate()/train_or_load() fonksiyonları
olduğu gibi kullanılır -> mantık burada tekrar yazılmaz, sadece birleştirilir.
"""

import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import mf
import ncf
import svd
from content_based import load_data as load_movies_with_tags, build_tfidf_matrix
from collaborative_filtering import build_matrix
from hybrid import content_scores, cf_scores, min_max_normalize, ALPHA, MIN_RATINGS

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_DIR = "../dataset/ml-latest-small"
K = 10
RELEVANCE_THRESHOLD = 4.0  # test setinde bu puanın üstü "kullanıcı gerçekten sevdi" sayılır
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAMPLE_USERS = 100  # 610 kullanıcının tamamını değerlendirmek yavaş olur, örneklem alınır


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_cf_rating_prediction(train_ratings, test_ratings, X, movie_mapper, movie_inv_mapper):
    # her kullanıcı için test filmlerine item-based CF ile puan tahmini yap, gerçek puanla kıyasla
    y_true, y_pred = [], []
    for user_id, group in test_ratings.groupby("userId"):
        preds = cf_scores(user_id, train_ratings, X, movie_mapper, movie_inv_mapper)
        for mid, actual in zip(group["movieId"], group["rating"]):
            if mid in preds:
                y_true.append(actual)
                y_pred.append(preds[mid])
    return rmse(y_true, y_pred), mae(y_true, y_pred), len(y_true)


def precision_recall_at_k(score_fn, train_ratings, test_ratings, user_ids, well_known, k=K, threshold=RELEVANCE_THRESHOLD):
    # score_fn(user_id) -> {movieId: score} dondurmeli
    precisions, recalls = [], []
    for user_id in user_ids:
        relevant = set(test_ratings[
            (test_ratings["userId"] == user_id) & (test_ratings["rating"] >= threshold)
        ]["movieId"])
        if not relevant:
            continue

        scores = score_fn(user_id)
        watched = set(train_ratings[train_ratings["userId"] == user_id]["movieId"])
        # az puanlanan filmler (tek kullanicidan gelen asiri/gurultulu skorlar) elenir -> hybrid.py ile ayni mantik
        candidates = {mid: s for mid, s in scores.items() if mid not in watched and mid in well_known}
        top_k = sorted(candidates, key=candidates.get, reverse=True)[:k]

        hit = len(set(top_k) & relevant)
        precisions.append(hit / k)
        recalls.append(hit / len(relevant))

    return float(np.mean(precisions)), float(np.mean(recalls)), len(precisions)


def main():
    movies = load_movies_with_tags()
    ratings = pd.read_csv(f"{DATASET_DIR}/ratings.csv")

    train_ratings, test_ratings = train_test_split(ratings, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # test setindeki filmler train'de de görülmüş olmalı, aksi halde model onları hiç tanımaz
    train_movie_ids = set(train_ratings["movieId"])
    test_ratings = test_ratings[test_ratings["movieId"].isin(train_movie_ids)]

    # degerlendirmeyi bir kullanici orneklemine indir -> TUM modeller/metrikler ayni orneklemle olcsun (adil kiyas)
    rng = np.random.RandomState(RANDOM_STATE)
    all_users = train_ratings["userId"].unique()
    sample_users = rng.choice(all_users, size=min(SAMPLE_USERS, len(all_users)), replace=False)
    test_ratings = test_ratings[test_ratings["userId"].isin(sample_users)]

    rating_counts = train_ratings.groupby("movieId").size()
    well_known = set(rating_counts[rating_counts >= MIN_RATINGS].index)

    # ---- content-based / item-based CF / hybrid icin gereken yapilar ----
    tfidf_matrix = build_tfidf_matrix(movies)
    X, cf_movie_mapper, cf_movie_inv_mapper = build_matrix(train_ratings)

    # ---- SVD / MF / NCF icin ortak mapper (uc model de ayni fonksiyonu kullaniyor) ----
    user_mapper, movie_mapper, movie_inv_mapper = svd.build_mappers(ratings)

    svd_predictions = svd.train_or_load(train_ratings, user_mapper, movie_mapper)

    global_mean = float(train_ratings["rating"].mean())
    mf_loader = DataLoader(mf.RatingsDataset(train_ratings, user_mapper, movie_mapper), batch_size=mf.BATCH_SIZE, shuffle=True)
    mf_model = mf.train_or_load(len(user_mapper), len(movie_mapper), global_mean, mf_loader)

    ncf_loader = DataLoader(ncf.RatingsDataset(train_ratings, user_mapper, movie_mapper), batch_size=ncf.BATCH_SIZE, shuffle=True)
    ncf_model = ncf.train_or_load(len(user_mapper), len(movie_mapper), ncf_loader)

    # ---- RMSE / MAE ----
    print(f"\n===== RMSE / MAE ({len(sample_users)} kullanıcı örneklemi) =====")

    rmse_cf, mae_cf, n_cf = evaluate_cf_rating_prediction(train_ratings, test_ratings, X, cf_movie_mapper, cf_movie_inv_mapper)
    print(f"{'Item-Based CF':15s} RMSE: {rmse_cf:.4f}   MAE: {mae_cf:.4f}   (n={n_cf})")

    rmse_svd, mae_svd, n_svd = svd.evaluate(test_ratings, train_movie_ids, svd_predictions, user_mapper, movie_mapper)
    print(f"{'SVD':15s} RMSE: {rmse_svd:.4f}   MAE: {mae_svd:.4f}   (n={n_svd})")

    rmse_mf, mae_mf, n_mf = mf.evaluate(mf_model, test_ratings, user_mapper, movie_mapper)
    print(f"{'MF':15s} RMSE: {rmse_mf:.4f}   MAE: {mae_mf:.4f}   (n={n_mf})")

    rmse_ncf, mae_ncf, n_ncf = ncf.evaluate(ncf_model, test_ratings, user_mapper, movie_mapper)
    print(f"{'NCF':15s} RMSE: {rmse_ncf:.4f}   MAE: {mae_ncf:.4f}   (n={n_ncf})")

    # ---- Precision@K / Recall@K icin her modele bir score_fn(user_id) -> {movieId: skor} ----
    def content_score_fn(user_id):
        scores = content_scores(user_id, train_ratings, movies, tfidf_matrix)
        return dict(zip(movies["movieId"], scores))

    def cf_score_fn(user_id):
        return cf_scores(user_id, train_ratings, X, cf_movie_mapper, cf_movie_inv_mapper)

    def hybrid_score_fn(user_id):
        c = min_max_normalize(content_scores(user_id, train_ratings, movies, tfidf_matrix))
        cf_dict = cf_scores(user_id, train_ratings, X, cf_movie_mapper, cf_movie_inv_mapper)
        f = min_max_normalize(movies["movieId"].map(cf_dict).fillna(0).values)
        combined = ALPHA * f + (1 - ALPHA) * c
        return dict(zip(movies["movieId"], combined))

    def svd_score_fn(user_id):
        if user_id not in user_mapper:
            return {}
        row = svd_predictions[user_mapper[user_id]]
        return {movie_inv_mapper[i]: row[i] for i in range(len(movie_inv_mapper))}

    @torch.no_grad()
    def mf_score_fn(user_id):
        if user_id not in user_mapper:
            return {}
        movie_ids = list(movie_mapper.keys())
        u = torch.tensor([user_mapper[user_id]] * len(movie_ids), dtype=torch.long)
        m = torch.tensor([movie_mapper[mid] for mid in movie_ids], dtype=torch.long)
        preds = mf_model(u, m).numpy()
        return dict(zip(movie_ids, preds))

    @torch.no_grad()
    def ncf_score_fn(user_id):
        if user_id not in user_mapper:
            return {}
        movie_ids = list(movie_mapper.keys())
        u = torch.tensor([user_mapper[user_id]] * len(movie_ids), dtype=torch.long)
        m = torch.tensor([movie_mapper[mid] for mid in movie_ids], dtype=torch.long)
        preds = ncf_model(u, m).numpy()
        return dict(zip(movie_ids, preds))

    print(f"\n===== Precision@{K} / Recall@{K} ({len(sample_users)} kullanıcı örneklemi) =====")
    for name, score_fn in [
        ("Content-Based", content_score_fn),
        ("Item-Based CF", cf_score_fn),
        ("SVD", svd_score_fn),
        ("MF", mf_score_fn),
        ("NCF", ncf_score_fn),
        ("Hybrid", hybrid_score_fn),
    ]:
        p, r, n_users = precision_recall_at_k(score_fn, train_ratings, test_ratings, sample_users, well_known)
        print(f"{name:15s} Precision@{K}: {p:.4f}   Recall@{K}: {r:.4f}   (n={n_users} kullanıcı)")


if __name__ == "__main__":
    main()
