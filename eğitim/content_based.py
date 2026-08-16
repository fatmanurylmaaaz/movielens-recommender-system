"""
Content Based Recommender - MovieLens (ml-latest-small)

Bir filmin içeriğine (tür/genre + kullanıcı etiketleri/tag) bakarak ona
benzer filmleri önerir. Kullanıcı geçmişine bakmaz; sadece "bu filme
benzeyen filmler" mantığı.
"""

import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_DIR = "../dataset/ml-latest-small"
TOP_N = 10


def load_data():
    # movies.csv: movieId, title, genres (örn: "Adventure|Animation|Comedy")
    # tags.csv: userId, movieId, tag, timestamp (kullanıcıların filme yazdığı serbest etiketler)
    movies = pd.read_csv(f"{DATASET_DIR}/movies.csv")
    tags = pd.read_csv(f"{DATASET_DIR}/tags.csv")

    # her film için tüm kullanıcı etiketlerini tek bir metinde topla
    tags_per_movie = tags.groupby("movieId")["tag"].apply(lambda t: " ".join(t.astype(str))).reset_index()
    movies = movies.merge(tags_per_movie, on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("")
    return movies


def build_tfidf_matrix(movies: pd.DataFrame):
    # genres + tags birleşimi -> "content soup": türler ile birlikte kullanıcı
    # etiketleri de (örn. "dark twist ending") benzerliğe katkı sağlar
    genres_text = movies["genres"].str.replace("|", " ", regex=False)
    content_soup = genres_text + " " + movies["tag"]
    tfidf = TfidfVectorizer(stop_words="english")
    return tfidf.fit_transform(content_soup)


def build_similarity_matrix(tfidf_matrix):
    # her film çifti arasındaki tür benzerliğini (0-1 arası) hesaplar
    return cosine_similarity(tfidf_matrix)


def recommend(movies: pd.DataFrame, similarity: "np.ndarray", title: str, n: int = TOP_N) -> pd.DataFrame:
    matches = movies.index[movies["title"].str.contains(title, case=False, regex=False)]
    if len(matches) == 0:
        raise ValueError(f"'{title}' ile eşleşen film bulunamadı")
    idx = matches[0]

    # seçilen filmin diğer tüm filmlerle benzerlik skorları
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:n]  # filmin kendisini çıkar

    result_idx = [i for i, _ in scores]
    result = movies.iloc[result_idx][["title", "genres"]].copy()
    result["similarity"] = [round(s, 3) for _, s in scores]
    return result


def main():
    movies = load_data()
    tfidf_matrix = build_tfidf_matrix(movies)
    similarity = build_similarity_matrix(tfidf_matrix)

    for sample_title in ["Toy Story", "Matrix"]:
        print(f"===== '{sample_title}' benzeri {TOP_N} film =====")
        print(recommend(movies, similarity, sample_title).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
