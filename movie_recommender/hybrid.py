"""Kullanici profili + agirlikli hybrid oneri (scripts/20-22'nin paketlenmis hali)."""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from . import config


def build_user_profile(user_id, ratings, movie_id_to_row, tfidf_matrix):
    """Kullanicinin puanladigi filmlerin TF-IDF vektorlerinin rating-agirlikli ortalamasi."""
    user_ratings = ratings[ratings["userId"] == user_id]
    rows, weights = [], []
    for mid, r in zip(user_ratings["movieId"], user_ratings["rating"]):
        if mid in movie_id_to_row:
            rows.append(movie_id_to_row[mid])
            weights.append(r)

    if not rows:
        return None

    weights = np.array(weights)
    profile = tfidf_matrix[rows].T.dot(weights) / weights.sum()
    return normalize(profile.reshape(1, -1)).flatten()


def build_all_user_profiles(ratings, movies, tfidf_matrix):
    movie_id_to_row = {mid: i for i, mid in enumerate(movies["movieId"])}
    profiles = {}
    for uid in ratings["userId"].unique():
        profile = build_user_profile(uid, ratings, movie_id_to_row, tfidf_matrix)
        if profile is not None:
            profiles[uid] = profile
    return profiles


def min_max_normalize(scores):
    rng = scores.max() - scores.min()
    return (scores - scores.min()) / rng if rng > 0 else np.zeros_like(scores)


def recommend_hybrid(
    user_id, movies, tfidf_matrix, predictions, user_mapper, movie_mapper,
    user_profiles, all_ratings, train_movies, well_known_movies,
    k=10, alpha=config.HYBRID_ALPHA,
):
    """CF ve content-based skorlarini normalize edip agirlikli birlestirir.
    Bir film train'de yoksa sadece content'e, content_text'i yoksa sadece CF'ye dusulur."""
    if user_id not in user_mapper or user_id not in user_profiles:
        return []

    user_idx = user_mapper[user_id]
    cf_norm = min_max_normalize(predictions[user_idx])
    content_norm = min_max_normalize(
        cosine_similarity(user_profiles[user_id].reshape(1, -1), tfidf_matrix).flatten()
    )

    already_watched = set(all_ratings[all_ratings["userId"] == user_id]["movieId"])
    has_content = (movies["content_text"] != "").values
    movie_id_to_row = {mid: i for i, mid in enumerate(movies["movieId"])}

    results = []
    for movie_id, row in movie_id_to_row.items():
        if movie_id in already_watched or movie_id not in well_known_movies:
            continue

        cf_ok = movie_id in train_movies and movie_id in movie_mapper
        content_ok = has_content[row]
        if not cf_ok and not content_ok:
            continue

        if cf_ok and content_ok:
            score = alpha * cf_norm[movie_mapper[movie_id]] + (1 - alpha) * content_norm[row]
        elif cf_ok:
            score = cf_norm[movie_mapper[movie_id]]
        else:
            score = content_norm[row]

        results.append((movies.loc[row, "clean_title"], float(score)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]
