# Content-based oneri fonksiyonu: TF-IDF + cosine similarity

import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = Path(__file__).resolve().parent
movies = pd.read_csv(SCRIPT_DIR / "movies_processed.csv")
tfidf_matrix = sp.load_npz(SCRIPT_DIR / "tfidf_matrix.npz")


def find_movie(title_query, df):
    query = title_query.strip()

    def starts_with(q):
        return df[df["clean_title"].str.lower().str.startswith(q.lower())]

    # "Baslik, The (yil)" formatina uyum icin ayni donusum burada da uygulaniyor
    result = starts_with(query)
    if result.empty and query.lower().startswith("the "):
        result = starts_with(query[4:] + ", The")
    if not result.empty:
        return result

    contains = df[df["clean_title"].str.contains(query, case=False, regex=False, na=False)].copy()
    return contains.sort_values(by="clean_title", key=lambda s: s.str.len())


def recommend(title_query, k=10):
    matches = find_movie(title_query, movies)
    if matches.empty:
        print(f"'{title_query}' basligiyla eslesen film bulunamadi.")
        return

    query_idx = matches.index[0]
    matched_title = movies.loc[query_idx, "clean_title"]

    if movies.loc[query_idx, "content_text"] == "":
        print(f"'{matched_title}' icin ne genre ne de tag bilgisi var, oneri uretilemiyor.")
        return

    sim_scores = cosine_similarity(tfidf_matrix[query_idx], tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1]

    print(f"\n'{matched_title}' filmini sevdiysen, bunlari da begenebilirsin:")
    shown = 0
    for idx in top_indices:
        if idx == query_idx:
            continue
        title = movies.loc[idx, "clean_title"]
        genres = movies.loc[idx, "genre_text"]
        print(f"  - {title}  (benzerlik: {sim_scores[idx]:.3f}, tur: {genres})")
        shown += 1
        if shown == k:
            break


if __name__ == "__main__":
    for query in ["The Dark Knight", "Titanic", "The Matrix", "Notting Hill"]:
        recommend(query, k=5)
