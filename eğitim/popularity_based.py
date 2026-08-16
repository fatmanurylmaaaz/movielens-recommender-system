"""
Popularity Based Recommender - MovieLens (ml-latest-small)

Herkese aynı öneriyi sunan, kullanıcıya özel olmayan (non-personalized)
bir öneri sistemi. En çok oy alan / en yüksek puanlı filmleri önerir.
"""

import sys
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_DIR = "../dataset/ml-latest-small"
TOP_N = 10
MIN_VOTE_COUNT = 50  # ağırlıklı puanlama için minimum oy sayısı eşiği


def load_data():
    # movies.csv: movieId, title, genres
    # ratings.csv: userId, movieId, rating, timestamp
    movies = pd.read_csv(f"{DATASET_DIR}/movies.csv")
    ratings = pd.read_csv(f"{DATASET_DIR}/ratings.csv")
    return movies, ratings


def build_movie_stats(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    # her film için oy sayısı ve ortalama puanı hesapla
    stats = ratings.groupby("movieId").agg(
        vote_count=("rating", "count"),
        vote_average=("rating", "mean"),
    ).reset_index()
    # film bilgileriyle (title) birleştir
    return movies.merge(stats, on="movieId", how="inner")


def top_by_count(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    # en çok oy alan filmler -> düşük puanlı ama popüler filmler öne çıkabilir
    return df.sort_values("vote_count", ascending=False).head(n)[
        ["title", "vote_count", "vote_average"]
    ]


def top_by_average(df: pd.DataFrame, n: int = TOP_N, min_votes: int = MIN_VOTE_COUNT) -> pd.DataFrame:
    # az oylanan filmleri ele, kalanları ortalama puana göre sırala
    filtered = df[df["vote_count"] >= min_votes]
    return filtered.sort_values("vote_average", ascending=False).head(n)[
        ["title", "vote_count", "vote_average"]
    ]


def weighted_rating(df: pd.DataFrame, min_votes: int = MIN_VOTE_COUNT) -> pd.Series:
    # IMDB ağırlıklı puan formülü: WR = (v/(v+m))*R + (m/(v+m))*C
    # oy sayısı (v) arttıkça filmin kendi ortalamasına (R) güvenir,
    # azsa genel ortalamaya (C) doğru çeker -> az oylu filmler haksız öne çıkmaz
    v = df["vote_count"]
    R = df["vote_average"]
    C = df["vote_average"].mean()
    m = min_votes
    return (v / (v + m)) * R + (m / (v + m)) * C


def top_by_weighted_score(df: pd.DataFrame, n: int = TOP_N, min_votes: int = MIN_VOTE_COUNT) -> pd.DataFrame:
    df = df.copy()
    df["weighted_score"] = weighted_rating(df, min_votes)
    return df.sort_values("weighted_score", ascending=False).head(n)[
        ["title", "vote_count", "vote_average", "weighted_score"]
    ]


def main():
    movies, ratings = load_data()
    movie_stats = build_movie_stats(movies, ratings)

    print(f"Toplam film sayısı: {movies.shape[0]}")
    print(f"Toplam rating sayısı: {ratings.shape[0]}")
    print(f"Puanlanmış film sayısı: {movie_stats.shape[0]}\n")

    print(f"===== En Çok Oy Alan {TOP_N} Film =====")
    print(top_by_count(movie_stats).to_string(index=False))

    print(f"\n===== En Yüksek Ortalama Puanlı {TOP_N} Film (min {MIN_VOTE_COUNT} oy) =====")
    print(top_by_average(movie_stats).to_string(index=False))

    print(f"\n===== Ağırlıklı Puana Göre En İyi {TOP_N} Film (IMDB formülü) =====")
    print(top_by_weighted_score(movie_stats).to_string(index=False))


if __name__ == "__main__":
    main()
