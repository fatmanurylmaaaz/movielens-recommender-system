# Hybrid oneri: CF + content-based agirlikli birlesim, cold-start fallback ile

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

# Adim 5'teki (05_popular_movies.py) dersle ayni sorun: cok az puanlanmis filmler
# content skorunda rastlantisal olarak yapay yuksek benzerlik yakalayabiliyor.
# Bu yuzden az taninan (guvenilmez) filmleri oneri havuzundan cikariyoruz.
MIN_RATINGS = 5
rating_counts = all_ratings.groupby("movieId").size()
well_known_movies = set(rating_counts[rating_counts >= MIN_RATINGS].index)

# movie_mapper indeksi (CF) ile movies_processed.csv satir indeksi (content) farkli
# siralamada olabilir, ikisini movieId uzerinden eslestirmemiz lazim
movie_id_to_content_row = {mid: i for i, mid in enumerate(movies["movieId"])}
movie_id_to_cf_col = mappers["movie_mapper"]

has_content = (movies["content_text"] != "").values  # movies_processed.csv sirasinda


def min_max_normalize(scores):
    rng = scores.max() - scores.min()
    if rng == 0:
        return np.zeros_like(scores)
    return (scores - scores.min()) / rng


def recommend_hybrid(user_id, k=10, alpha=0.5):
    if user_id not in mappers["user_mapper"] or user_id not in user_profiles:
        print(f"userId {user_id} icin yeterli veri yok.")
        return

    user_idx = mappers["user_mapper"][user_id]
    cf_scores_raw = predictions[user_idx]  # movie_mapper sirasinda (9724,)
    content_scores_raw = cosine_similarity(
        user_profiles[user_id].reshape(1, -1), tfidf_matrix
    ).flatten()  # movies_processed.csv sirasinda (9724,)

    cf_norm = min_max_normalize(cf_scores_raw)
    content_norm = min_max_normalize(content_scores_raw)

    already_watched = set(all_ratings[all_ratings["userId"] == user_id]["movieId"])

    results = []
    for movie_id, content_row in movie_id_to_content_row.items():
        if movie_id in already_watched:
            continue
        if movie_id not in well_known_movies:
            continue

        cf_reliable = movie_id in train_movies and movie_id in movie_id_to_cf_col
        content_reliable = has_content[content_row]

        if not cf_reliable and not content_reliable:
            continue  # ikisi de guvenilmezse oneri uretme

        if cf_reliable and content_reliable:
            cf_val = cf_norm[movie_id_to_cf_col[movie_id]]
            content_val = content_norm[content_row]
            score = alpha * cf_val + (1 - alpha) * content_val
            source = "hybrid"
        elif cf_reliable:
            score = cf_norm[movie_id_to_cf_col[movie_id]]
            source = "sadece CF (content yok)"
        else:
            score = content_norm[content_row]
            source = "sadece content (train'de yok)"

        results.append((movie_id, score, source))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== userId {user_id} icin HYBRID oneriler (alpha={alpha}) ===")
    for movie_id, score, source in results[:k]:
        title = movies.loc[movies["movieId"] == movie_id, "clean_title"].values[0]
        print(f"  - {title}  (skor: {score:.3f}, kaynak: {source})")


if __name__ == "__main__":
    # alpha=0.9: 26_tune_alpha.py'de Precision@10 ile secilen en iyi deger
    # (saf CF'den bile hafifce daha iyi cikti, alpha=0.5 cok kotu bir secimdi)
    recommend_hybrid(1, k=10, alpha=0.9)
