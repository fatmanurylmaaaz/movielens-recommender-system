"""Tum pipeline'i tek bir sinifta toplayan orkestratork (scripts/01-26'nin ozeti)."""

import pickle
import numpy as np
from . import config, data, preprocessing, content_based, collaborative, hybrid, evaluate


class MovieRecommender:
    """Egitildikten sonra content-based, collaborative filtering ve hybrid
    onerileri tek bir arayuzden sunan model. scikit-learn tarzinda fit() ile
    egitilir, recommend_* metotlariyla sorgulanir."""

    def __init__(self):
        self.movies = None
        self.all_ratings = None
        self.train_ratings = None
        self.test_ratings = None
        self.tfidf_matrix = None
        self.predictions = None
        self.user_mapper = None
        self.movie_mapper = None
        self.movie_inv_mapper = None
        self.user_profiles = None
        self.train_movies = None
        self.well_known_movies = None

    def fit(self, use_cache=True):
        cache_file = config.CACHE_DIR / "model.pkl"
        if use_cache and cache_file.exists():
            with open(cache_file, "rb") as f:
                self.__dict__.update(pickle.load(f))
            return self

        movies_raw = data.load_movies()
        tags = data.load_tags()
        self.all_ratings = data.load_ratings()

        self.movies = preprocessing.preprocess_movies(movies_raw, tags)
        _, self.tfidf_matrix = content_based.build_tfidf(self.movies)

        self.user_mapper, self.movie_mapper, self.movie_inv_mapper = collaborative.build_mappers(
            self.all_ratings
        )
        self.train_ratings, self.test_ratings = collaborative.split_ratings(self.all_ratings)
        self.train_movies = set(self.train_ratings["movieId"])

        self.predictions, _ = collaborative.train_svd(
            self.train_ratings, self.user_mapper, self.movie_mapper
        )

        self.user_profiles = hybrid.build_all_user_profiles(
            self.all_ratings, self.movies, self.tfidf_matrix
        )

        rating_counts = self.all_ratings.groupby("movieId").size()
        self.well_known_movies = set(rating_counts[rating_counts >= config.MIN_RATINGS].index)

        if use_cache:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(self.__dict__, f)

        return self

    def recommend_content(self, movie_title, k=10):
        return content_based.recommend_similar(movie_title, self.movies, self.tfidf_matrix, k=k)

    def recommend_cf(self, user_id, k=10):
        return collaborative.recommend_for_user(
            user_id, self.predictions, self.user_mapper, self.movie_inv_mapper,
            self.movies, self.all_ratings, k=k,
        )

    def recommend_hybrid(self, user_id, k=10, alpha=config.HYBRID_ALPHA):
        return hybrid.recommend_hybrid(
            user_id, self.movies, self.tfidf_matrix, self.predictions,
            self.user_mapper, self.movie_mapper, self.user_profiles,
            self.all_ratings, self.train_movies, self.well_known_movies,
            k=k, alpha=alpha,
        )

    def evaluate_cf(self):
        """CF modelinin test setindeki RMSE/MAE degerlerini dondurur (sadece train'de
        gorulen filmler uzerinden, adim 5'teki gerekceyle)."""
        test = self.test_ratings.copy()
        test = test[test["movieId"].isin(self.train_movies)]
        test["user_idx"] = test["userId"].map(self.user_mapper)
        test["movie_idx"] = test["movieId"].map(self.movie_mapper)
        pred = self.predictions[test["user_idx"], test["movie_idx"]]
        return evaluate.rmse(test["rating"], pred), evaluate.mae(test["rating"], pred)
