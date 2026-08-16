"""Genre/tag temizligi ve content_text olusturma (scripts/07-09'un paketlenmis hali)."""

import pandas as pd
from . import config


def parse_genres(genre_str):
    if genre_str == "(no genres listed)":
        return []
    return genre_str.split("|")


def add_genre_features(movies):
    movies = movies.copy()
    movies["genre_list"] = movies["genres"].apply(parse_genres)

    # TF-IDF'in "Sci-Fi" / "Film-Noir" gibi birlesik turleri ikiye bolmemesi icin
    # tirenleri kaldirip tek kelime haline getiriyoruz
    movies["genre_text"] = movies["genre_list"].apply(
        lambda genres: " ".join(g.replace("-", "").replace(" ", "").lower() for g in genres)
    )

    # title icindeki yili ayri bir sutuna cikar; yil bulunamazsa clean_title orijinal
    # title'a dussun (NaN kalirsa arama fonksiyonlari hata verir)
    extracted = movies["title"].str.extract(r"^(.*)\s\((\d{4})\)\s*$")
    movies["clean_title"] = extracted[0].fillna(movies["title"])
    movies["year"] = pd.to_numeric(extracted[1])
    return movies


def add_tag_features(movies, tags):
    movies = movies.copy()
    tags = tags.copy()

    tags["tag_clean"] = tags["tag"].str.lower().str.strip()
    tags = tags[~tags["tag_clean"].isin(config.NOISE_TAGS)]

    movie_tags = tags.groupby("movieId")["tag_clean"].apply(lambda s: " ".join(s)).reset_index()
    movie_tags.columns = ["movieId", "tag_text"]

    movies = movies.merge(movie_tags, on="movieId", how="left")
    movies["tag_text"] = movies["tag_text"].fillna("")
    return movies


def add_content_text(movies):
    movies = movies.copy()
    movies["genre_text"] = movies["genre_text"].fillna("")
    movies["tag_text"] = movies["tag_text"].fillna("")

    # genre_text 2 kez tekrarlanarak biraz daha fazla agirlik veriliyor
    # (genre her filmde guvenilir bilgi, tag'ler sadece bazi filmlerde var)
    movies["content_text"] = (
        movies["genre_text"] + " " + movies["genre_text"] + " " + movies["tag_text"]
    ).str.strip()
    return movies


def preprocess_movies(movies, tags):
    """movies.csv + tags.csv -> content_text dahil tam islenmis movies dataframe."""
    movies = add_genre_features(movies)
    movies = add_tag_features(movies, tags)
    movies = add_content_text(movies)
    return movies.reset_index(drop=True)
