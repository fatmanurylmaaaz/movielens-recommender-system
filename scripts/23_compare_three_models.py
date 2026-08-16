# CF, Content-based ve Hybrid modellerini ayni kosullar altinda kiyaslama

import importlib.util
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"


def load_module(filename):
    spec = importlib.util.spec_from_file_location(filename, SCRIPT_DIR / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


content_based = load_module("12_content_based_recommend")
movies = content_based.movies
tfidf_matrix = content_based.tfidf_matrix

predictions = np.load(SCRIPT_DIR / "predictions.npy")
with open(SCRIPT_DIR / "cf_mappers.pkl", "rb") as f:
    mappers = pickle.load(f)
with open(SCRIPT_DIR / "user_profiles.pkl", "rb") as f:
    user_profiles = pickle.load(f)

all_ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
train_ratings = pd.read_csv(SCRIPT_DIR / "ratings_train.csv")
train_movies = set(train_ratings["movieId"])

MIN_RATINGS = 5
rating_counts = all_ratings.groupby("movieId").size()
well_known_movies = set(rating_counts[rating_counts >= MIN_RATINGS].index)

movie_id_to_content_row = {mid: i for i, mid in enumerate(movies["movieId"])}
movie_id_to_cf_col = mappers["movie_mapper"]
has_content = (movies["content_text"] != "").values


def min_max_normalize(scores):
    rng = scores.max() - scores.min()
    return (scores - scores.min()) / rng if rng > 0 else np.zeros_like(scores)


def get_three_lists(user_id, k=5, alpha=0.5):
    user_idx = mappers["user_mapper"][user_id]
    cf_scores_raw = predictions[user_idx]
    content_scores_raw = cosine_similarity(
        user_profiles[user_id].reshape(1, -1), tfidf_matrix
    ).flatten()
    cf_norm = min_max_normalize(cf_scores_raw)
    content_norm = min_max_normalize(content_scores_raw)

    already_watched = set(all_ratings[all_ratings["userId"] == user_id]["movieId"])

    cf_only, content_only, hybrid = [], [], []
    for movie_id, content_row in movie_id_to_content_row.items():
        if movie_id in already_watched or movie_id not in well_known_movies:
            continue

        cf_reliable = movie_id in train_movies and movie_id in movie_id_to_cf_col
        content_reliable = has_content[content_row]

        if cf_reliable:
            cf_only.append((movie_id, cf_norm[movie_id_to_cf_col[movie_id]]))
        if content_reliable:
            content_only.append((movie_id, content_norm[content_row]))
        if cf_reliable and content_reliable:
            score = alpha * cf_norm[movie_id_to_cf_col[movie_id]] + (1 - alpha) * content_norm[content_row]
            hybrid.append((movie_id, score))

    cf_only.sort(key=lambda x: x[1], reverse=True)
    content_only.sort(key=lambda x: x[1], reverse=True)
    hybrid.sort(key=lambda x: x[1], reverse=True)
    return cf_only[:k], content_only[:k], hybrid[:k]


def title_of(movie_id):
    return movies.loc[movies["movieId"] == movie_id, "clean_title"].values[0]


USER_ID = 1
cf_list, content_list, hybrid_list = get_three_lists(USER_ID, k=5, alpha=0.5)

print(f"=== userId {USER_ID} icin 3 model karsilastirmasi ===\n")
print(f"{'CF-only':35s} | {'Content-only':35s} | {'Hybrid':35s}")
print("-" * 110)
for i in range(5):
    cf_title = title_of(cf_list[i][0])[:33] if i < len(cf_list) else ""
    content_title = title_of(content_list[i][0])[:33] if i < len(content_list) else ""
    hybrid_title = title_of(hybrid_list[i][0])[:33] if i < len(hybrid_list) else ""
    print(f"{cf_title:35s} | {content_title:35s} | {hybrid_title:35s}")

cf_ids = {m for m, _ in cf_list}
content_ids = {m for m, _ in content_list}
hybrid_ids = {m for m, _ in hybrid_list}

print(f"\nCF ile Hybrid ortak film sayisi: {len(cf_ids & hybrid_ids)}/5")
print(f"Content ile Hybrid ortak film sayisi: {len(content_ids & hybrid_ids)}/5")
print(f"CF ile Content ortak film sayisi: {len(cf_ids & content_ids)}/5")
