"""
Item-Based Collaborative Filtering - MovieLens (ml-latest-small)

Filmin içeriğine değil, kullanıcıların verdiği puanlara bakar:
"Bu filmi puanlayan kullanıcılar, hangi başka filmlere benzer puanlar vermiş?"
"""

import sys
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_DIR = "../dataset/ml-latest-small"
TOP_N = 10


def load_data():
    ratings = pd.read_csv(f"{DATASET_DIR}/ratings.csv")
    movies = pd.read_csv(f"{DATASET_DIR}/movies.csv")
    return ratings, movies


def build_matrix(ratings: pd.DataFrame):
    # satır = film, sütun = kullanıcı, deger = rating -> seyrek (sparse) matris
    # boş hücreler (puanlanmamış filmler) 0 kabul edilir, bellekten tasarruf icin csr_matrix kullanilir
    user_mapper = {uid: i for i, uid in enumerate(ratings["userId"].unique())}
    movie_mapper = {mid: i for i, mid in enumerate(ratings["movieId"].unique())}
    movie_inv_mapper = {i: mid for mid, i in movie_mapper.items()}

    user_index = ratings["userId"].map(user_mapper)
    movie_index = ratings["movieId"].map(movie_mapper)

    X = csr_matrix(
        (ratings["rating"], (movie_index, user_index)),
        shape=(len(movie_mapper), len(user_mapper)),
    )
    return X, movie_mapper, movie_inv_mapper


def find_movie(title_query: str, movies: pd.DataFrame) -> pd.DataFrame:
    # MovieLens basliklari "Baslik, The (yil)" formatinda -> "The X" aramasini da destekle
    query = title_query.strip()

    def starts_with(q):
        return movies[movies["title"].str.lower().str.startswith(q.lower())]

    result = starts_with(query)
    if result.empty and query.lower().startswith("the "):
        result = starts_with(query[4:] + ", The")
    if not result.empty:
        return result

    # basa denk gelen yoksa baslik icinde gecen eslesmelere bak, en kisa baslik en alakali olur
    contains = movies[movies["title"].str.contains(query, case=False, regex=False, na=False)].copy()
    return contains.sort_values(by="title", key=lambda s: s.str.len())


def recommend(movie_title: str, movies: pd.DataFrame, X, movie_mapper: dict, movie_inv_mapper: dict, k: int = TOP_N):
    matches = find_movie(movie_title, movies)
    if matches.empty:
        raise ValueError(f"'{movie_title}' ile eşleşen film bulunamadı")

    movie_id = matches["movieId"].iloc[0]
    matched_title = matches["title"].iloc[0]
    movie_idx = movie_mapper[movie_id]

    # kosinus benzerligine gore en yakin k+1 komsuyu bul (ilki filmin kendisi, atilacak)
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(X)
    distances, indices = model.kneighbors(X[movie_idx], n_neighbors=k + 1)

    neighbor_ids = [movie_inv_mapper[i] for i in indices.flatten()[1:]]
    similarities = 1 - distances.flatten()[1:]  # cosine distance -> similarity

    result = movies[movies["movieId"].isin(neighbor_ids)][["title"]].copy()
    result = result.set_index(movies.loc[movies["movieId"].isin(neighbor_ids)].index)
    id_to_similarity = dict(zip(neighbor_ids, similarities))
    result["similarity"] = movies.loc[result.index, "movieId"].map(id_to_similarity)
    result = result.sort_values("similarity", ascending=False)

    return matched_title, result


def main():
    ratings, movies = load_data()
    X, movie_mapper, movie_inv_mapper = build_matrix(ratings)

    for sample_title in ["Toy Story", "The Dark Knight"]:
        matched_title, result = recommend(sample_title, movies, X, movie_mapper, movie_inv_mapper)
        print(f"===== '{matched_title}' beğenenlerin sevdiği {TOP_N} film =====")
        print(result.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
